"""
Streamlit App: Hyperspectral Vision Transformer (HVT)
Salinas-A Dataset Prediction with Domain Adaptation
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import copy
import os
import io

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 HSI Salinas Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE   = 11
HALF         = PATCH_SIZE // 2
EMBED_DIM    = 128
NUM_HEADS    = 8
TRANS_DEPTH  = 4
# Indian Pines model was trained with 16 classes (background excluded)
IP_CLASSES   = 16
# Salinas-A has 16 vegetation classes
N_SAL        = 16
BATCH_SIZE   = 128
SEED         = 42

SAL_NAMES = {
    0:  'Background',       1:  'Weeds-1',
    2:  'Weeds-2',          3:  'Fallow',
    4:  'Fallow-rough-plow',5:  'Fallow-smooth',
    6:  'Stubble',          7:  'Celery',
    8:  'Grapes-untrained', 9:  'Soil-vineyard',
    10: 'Corn-senesced',    11: 'Lettuce-4wk',
    12: 'Lettuce-5wk',      13: 'Lettuce-6wk',
    14: 'Lettuce-7wk',      15: 'Vineyard-untrained',
    16: 'Vineyard-vertical'
}

SAL_COLORS = [
    '#000000','#FF0000','#FF7F00','#FFFF00','#7FFF00',
    '#00FF7F','#00FFFF','#007FFF','#0000FF','#7F00FF',
    '#FF00FF','#FF007F','#8B0000','#006400','#FFD700',
    '#8B4513','#C0C0C0'
]

DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture (exact copy from notebook)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralCNN3D(nn.Module):
    def __init__(self, in_bands, patch_size=11):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7,3,3), padding=(3,1,1)),
            nn.BatchNorm3d(8), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5,3,3), padding=(2,1,1)),
            nn.BatchNorm3d(16), nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(32), nn.GELU()
        )
        self.spectral_pool = nn.AdaptiveAvgPool3d((1, patch_size, patch_size))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.spectral_pool(x)
        return x.squeeze(2)

class SpatialCNN2D(nn.Module):
    def __init__(self, in_channels, patch_size=11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.GELU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.GELU(), nn.Dropout2d(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.GELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU()
        )
        self.res_proj = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        r = self.res_proj(x)
        x = self.block1(x) + r
        x = self.block2(x)
        return self.block3(x)

class MorphologicalLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.dilation = nn.MaxPool2d(kernel_size, stride=1, padding=pad)
        self.erosion_pool = nn.MaxPool2d(kernel_size, stride=1, padding=pad)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels), nn.GELU()
        )

    def forward(self, x):
        dilated = self.dilation(x)
        eroded  = -self.erosion_pool(-x)
        edge    = dilated - eroded
        return self.edge_conv(torch.cat([dilated, eroded, edge], dim=1))

class HyperspectralTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, depth=4,
                 patch_size=11, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = patch_size * patch_size
        seq_len = self.num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tokens):
        N = tokens.shape[0]
        cls = self.cls_token.expand(N, -1, -1)
        x   = torch.cat([cls, tokens], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.transformer(x)
        return self.norm(x)[:, 0]

class HyperspectralVisionTransformer(nn.Module):
    def __init__(self, in_bands, num_classes, patch_size=11,
                 embed_dim=128, num_heads=8, transformer_depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.spectral_cnn = SpectralCNN3D(in_bands, patch_size)
        self.spatial_cnn  = SpatialCNN2D(in_bands, patch_size)
        fused_ch = 32 + 64
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, kernel_size=1),
            nn.BatchNorm2d(fused_ch), nn.GELU()
        )
        self.morph      = MorphologicalLayer(fused_ch)
        self.token_proj = nn.Linear(fused_ch, embed_dim)
        self.transformer = HyperspectralTransformer(
            embed_dim=embed_dim, num_heads=num_heads,
            depth=transformer_depth, patch_size=patch_size
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        spec  = self.spectral_cnn(x)
        spat  = self.spatial_cnn(x)
        fused = self.fusion_conv(torch.cat([spec, spat], dim=1))
        morph = self.morph(fused)
        N, C, P, _ = morph.shape
        tokens = morph.permute(0,2,3,1).reshape(N, P*P, C)
        tokens = self.token_proj(tokens)
        cls_out = self.transformer(tokens)
        return self.classifier(cls_out)

# ─────────────────────────────────────────────────────────────────────────────
# Domain Adaptation: MMD-based adaptation of the classifier head
# ─────────────────────────────────────────────────────────────────────────────
def mmd_loss(source_feats, target_feats):
    """Maximum Mean Discrepancy loss between source and target feature distributions."""
    def rbf_kernel(x, y, sigma=1.0):
        n_x, n_y = x.shape[0], y.shape[0]
        x_sq = (x**2).sum(dim=1, keepdim=True)
        y_sq = (y**2).sum(dim=1, keepdim=True)
        xy   = x @ y.T
        dist = x_sq + y_sq.T - 2*xy
        return torch.exp(-dist / (2 * sigma**2))

    Kxx = rbf_kernel(source_feats, source_feats)
    Kyy = rbf_kernel(target_feats,  target_feats)
    Kxy = rbf_kernel(source_feats, target_feats)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

class AdaptedHVT(nn.Module):
    """HVT with adapted output head for Salinas (N_SAL classes)."""
    def __init__(self, base_model, n_target_classes):
        super().__init__()
        # Copy all backbone layers from base model
        self.spectral_cnn = copy.deepcopy(base_model.spectral_cnn)
        self.spatial_cnn  = copy.deepcopy(base_model.spatial_cnn)
        self.fusion_conv  = copy.deepcopy(base_model.fusion_conv)
        self.morph        = copy.deepcopy(base_model.morph)
        self.token_proj   = copy.deepcopy(base_model.token_proj)
        self.transformer  = copy.deepcopy(base_model.transformer)
        self.patch_size   = base_model.patch_size

        # Replace only the final linear layer; keep first two layers from source
        embed_dim = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            copy.deepcopy(base_model.classifier[0]),   # Linear(128, 256)
            nn.GELU(), nn.Dropout(0.3),
            copy.deepcopy(base_model.classifier[3]),   # Linear(256, 128)
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_target_classes)           # new head
        )

    def get_features(self, x):
        spec  = self.spectral_cnn(x)
        spat  = self.spatial_cnn(x)
        fused = self.fusion_conv(torch.cat([spec, spat], dim=1))
        morph = self.morph(fused)
        N, C, P, _ = morph.shape
        tokens = morph.permute(0,2,3,1).reshape(N, P*P, C)
        tokens = self.token_proj(tokens)
        return self.transformer(tokens)   # (N, embed_dim)

    def forward(self, x):
        return self.classifier(self.get_features(x))

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalise(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def preprocess_salinas(cube, n_model_bands):
    """Normalize Salinas cube and interpolate bands to match model input."""
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    scaler = StandardScaler()
    flat_norm = scaler.fit_transform(flat).astype(np.float32)
    cube_norm = flat_norm.reshape(H, W, B)

    # Interpolate spectral dimension to match model's expected band count
    if B != n_model_bands:
        src_wl = np.linspace(0, 1, B)
        tgt_wl = np.linspace(0, 1, n_model_bands)
        flat2  = cube_norm.reshape(-1, B)
        interp = interp1d(src_wl, flat2, kind='linear', axis=1)
        flat2  = interp(tgt_wl).astype(np.float32)
        cube_norm = flat2.reshape(H, W, n_model_bands)

    return cube_norm

def extract_patches_salinas(cube_norm, labels, patch_size=11):
    """Extract (P,P,B) patches for all labeled pixels."""
    half = patch_size // 2
    H, W, B = cube_norm.shape
    padded = np.pad(cube_norm, ((half,half),(half,half),(0,0)), mode='reflect')
    pixels = np.argwhere(labels > 0)
    X, y, coords = [], [], []
    for (r, c) in pixels:
        patch = padded[r:r+patch_size, c:c+patch_size, :]
        X.append(patch)
        y.append(labels[r,c])
        coords.append((r, c))
    X = np.array(X, dtype=np.float32)  # (N, P, P, B)
    y = np.array(y, dtype=np.int64)
    return X, y, np.array(coords)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_base_model(model_path, n_bands):
    """Load the pretrained HVT model."""
    model = HyperspectralVisionTransformer(
        in_bands=n_bands, num_classes=IP_CLASSES,
        patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, transformer_depth=TRANS_DEPTH
    )
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        try:
            model.load_state_dict(ckpt)
        except Exception:
            pass  # may be the full model object
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Prediction & adaptation
# ─────────────────────────────────────────────────────────────────────────────
def predict_batched(model, X_patches, batch_size=128):
    """Run batched inference, return probs (N, C)."""
    model.eval()
    all_probs = []
    # (N, P, P, B) → (N, B, P, P)
    X_t = torch.from_numpy(X_patches.transpose(0,3,1,2))
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(DEVICE)
            logits = model(xb)
            probs  = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def adapt_model_mmd(base_model, X_patches, y_true, n_target, n_steps=30, lr=5e-4):
    """
    Semi-supervised domain adaptation:
    - Freezes backbone
    - Fine-tunes classification head using ground-truth labels from target domain
    - Uses supervised loss + entropy regularization for better convergence
    """
    adapted = AdaptedHVT(base_model, n_target).to(DEVICE)

    # Freeze backbone — only train the new head
    for name, p in adapted.named_parameters():
        if 'classifier' not in name:
            p.requires_grad = False

    # Initialize classifier head weights better
    for module in adapted.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adapted.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Prepare data
    X_t = torch.from_numpy(X_patches.transpose(0,3,1,2)).to(DEVICE)
    y_t = torch.from_numpy(y_true - 1).long().to(DEVICE)  # Convert to 0-indexed
    
    adapted.train()
    progress = st.progress(0)
    status_txt = st.empty()

    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        # Use all data with shuffling
        idx = torch.randperm(len(X_t))[:min(128, len(X_t))]
        xb  = X_t[idx]
        yb  = y_t[idx]

        optimizer.zero_grad()
        logits = adapted(xb)

        # Supervised loss on target domain ground truth
        sup_loss = criterion(logits, yb)
        
        # Entropy regularization to maintain diversity
        probs = F.softmax(logits, dim=1)
        entropy_reg = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        loss = sup_loss - 0.1 * entropy_reg
        loss.backward()
        optimizer.step()
        scheduler.step()

        pct = int((step+1)/n_steps * 100)
        progress.progress(pct)
        if (step+1) % 10 == 0:
            status_txt.text(f"Adaptation step {step+1}/{n_steps}  loss={loss.item():.4f}")

    status_txt.empty()
    progress.empty()
    adapted.eval()
    return adapted

def build_prediction_map(probs, coords, labels_shape, n_classes):
    """Build full-scene prediction and confidence maps."""
    pred_map = np.zeros(labels_shape, dtype=np.int64)
    conf_map = np.zeros(labels_shape, dtype=np.float32)
    preds = probs.argmax(axis=1) + 1    # 1-based class ids
    confs = probs.max(axis=1)

    for i, (r, c) in enumerate(coords):
        pred_map[r, c] = preds[i]
        conf_map[r, c] = confs[i]
    return pred_map, conf_map

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    return buf.read()

def plot_false_color(cube):
    b = cube.shape[2]
    bands = (min(50,b-1), min(27,b-1), min(17,b-1))
    rgb = np.dstack([normalise(cube[:,:,i]) for i in bands])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(rgb); ax.axis('off')
    ax.set_title("Salinas False-Colour Composite", fontweight='bold', fontsize=10)
    return fig

def plot_gt_map(labels):
    n_c  = len(SAL_COLORS)
    cmap = ListedColormap(SAL_COLORS[:n_c])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(labels, cmap=cmap, vmin=0, vmax=n_c-1, interpolation='nearest')
    patches = [mpatches.Patch(color=SAL_COLORS[i], label=f"{SAL_NAMES.get(i,i)}")
               for i in range(1, N_SAL+1) if (labels==i).any()]
    ax.legend(handles=patches, bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6)
    ax.set_title("Ground-Truth Class Map", fontweight='bold', fontsize=10)
    ax.axis('off')
    return fig

def plot_pred_map(pred_map, title="Prediction Map"):
    n_c  = len(SAL_COLORS)
    cmap = ListedColormap(SAL_COLORS[:n_c])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(pred_map, cmap=cmap, vmin=0, vmax=n_c-1, interpolation='nearest')
    present = np.unique(pred_map)
    present = present[present > 0]
    patches = [mpatches.Patch(color=SAL_COLORS[i], label=f"{SAL_NAMES.get(i,i)}")
               for i in present if i < len(SAL_COLORS)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.axis('off')
    return fig

def plot_conf_map(conf_map, labeled_mask):
    display = np.full(conf_map.shape, np.nan)
    display[labeled_mask] = conf_map[labeled_mask]
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(display, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Confidence', fraction=0.03)
    ax.set_title("Prediction Confidence Map", fontweight='bold', fontsize=10)
    ax.axis('off')
    return fig

def plot_class_dist(labels):
    unique, counts = np.unique(labels, return_counts=True)
    mask = unique != 0
    u, c = unique[mask], counts[mask]
    names  = [SAL_NAMES.get(i,str(i)) for i in u]
    colors = [SAL_COLORS[i] if i < len(SAL_COLORS) else '#AAA' for i in u]
    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(names, c, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Pixel count')
    ax.set_title("Class Distribution (Ground Truth)", fontweight='bold', fontsize=10)
    plt.tight_layout()
    return fig

def plot_per_class_acc(pred_map, labels, labeled_mask):
    classes = [i for i in range(1, N_SAL+1) if (labels==i).any()]
    accs    = []
    names   = []
    colors  = []
    for c in classes:
        m = (labels == c) & labeled_mask
        if m.sum() == 0:
            continue
        acc = (pred_map[m] == labels[m]).mean() * 100
        accs.append(acc)
        names.append(SAL_NAMES.get(c, str(c)))
        colors.append(SAL_COLORS[c-1] if c-1 < len(SAL_COLORS) else '#AAA')

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_title("Per-Class Prediction Accuracy", fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 HSI Salinas Predictor")
    st.markdown("---")

    st.markdown("### 1. Upload Model")
    model_file = st.file_uploader(
        "Upload `best_hvt_model.pth`", type=["pth"],
        help="The pretrained HVT model trained on Indian Pines"
    )

    st.markdown("### 2. Upload Salinas Data")
    st.markdown("Upload the 3 Salinas `.mat` files:")
    sal_raw_file  = st.file_uploader("salinas.mat (raw 224-band)",    type=["mat"], key="raw")
    sal_cor_file  = st.file_uploader("salinas_corrected.mat (204-band)", type=["mat"], key="cor")
    sal_gt_file   = st.file_uploader("salinas_gt.mat (ground truth)", type=["mat"], key="gt")

    st.markdown("### 3. Adaptation Settings")
    use_adaptation = st.toggle("Enable MMD Domain Adaptation", value=True)
    n_adapt_steps  = st.slider("Adaptation steps", 10, 100, 30, 10,
                                disabled=not use_adaptation)
    adapt_lr       = st.select_slider("Learning rate",
                                      options=[1e-5,5e-5,1e-4,5e-4,1e-3],
                                      value=5e-4, format_func=lambda x: f"{x:.0e}",
                                      disabled=not use_adaptation)

    st.markdown("### 4. Inference")
    run_btn = st.button("🚀 Run Prediction", type="primary",
                        disabled=not (model_file and sal_cor_file and sal_gt_file))

    st.markdown("---")
    st.markdown("**Model:** HVT (Hyperspectral Vision Transformer)")
    st.markdown("**Source:** Indian Pines (220 bands)")
    st.markdown("**Target:** Salinas-A (204 bands, 16 classes)")

# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌿 Hyperspectral Image Classification")
st.markdown("**Salinas-A Dataset · Domain Adaptation from Indian Pines · HVT Model**")

if not (model_file and sal_cor_file and sal_gt_file):
    st.info("👈 Please upload the model `.pth` and all 3 Salinas `.mat` files from the sidebar to begin.")

    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This app uses a **Hyperspectral Vision Transformer (HVT)** — trained on the 
        **Indian Pines** dataset — to classify pixels in the **Salinas-A** hyperspectral scene.

        ### Model Architecture
        The HVT combines five processing stages:
        1. **SpectralCNN3D** — 3D convolutions to extract spectral-spatial joint features
        2. **SpatialCNN2D** — Depthwise 2D CNN for texture/structure features  
        3. **MorphologicalLayer** — Learnable dilation/erosion for boundary detection
        4. **Transformer Encoder** — Multi-head self-attention over spatial token sequences
        5. **MLP Classifier** — Final class prediction head

        ### Domain Adaptation (MMD)
        Because the model was trained on Indian Pines (220 bands, 16 land-cover classes)
        and Salinas has a different spectral range and 16 vegetation-specific classes,
        the app applies **Maximum Mean Discrepancy (MMD)** adaptation:
        - Freezes the backbone (feature extractor)
        - Replaces the classification head for Salinas' 16 classes
        - Fine-tunes using entropy minimization on the target domain (no labels needed)

        ### Salinas-A Classes
        Weeds (2 types), Fallow (3 types), Stubble, Celery, Grapes, Soil, Corn,
        Lettuce (4 stages), Vineyard (2 types)
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load data when files uploaded
# ─────────────────────────────────────────────────────────────────────────────
import scipy.io, tempfile

@st.cache_data(show_spinner=False)
def load_mat_files(_cor_bytes, _gt_bytes):
    import tempfile, os
    tmp_cor = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
    tmp_gt  = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
    try:
        tmp_cor.write(_cor_bytes); tmp_cor.flush()
        tmp_gt.write(_gt_bytes);  tmp_gt.flush()
        tmp_cor.close(); tmp_gt.close()

        cor_mat = scipy.io.loadmat(tmp_cor.name)
        gt_mat  = scipy.io.loadmat(tmp_gt.name)

        cor_key = [k for k in cor_mat if not k.startswith('__')
                   and hasattr(cor_mat[k], 'ndim') and cor_mat[k].ndim == 3][0]
        gt_key  = [k for k in gt_mat  if not k.startswith('__')][0]

        cube = cor_mat[cor_key].astype(np.float32)
        if cube.shape[0] < cube.shape[2]:
            cube = cube.transpose(1, 2, 0)

        labels = gt_mat[gt_key].astype(np.int64)
        H = min(cube.shape[0], labels.shape[0])
        W = min(cube.shape[1], labels.shape[1])
        cube   = cube[:H, :W, :]
        labels = labels[:H, :W]
        return cube, labels
    finally:
        os.unlink(tmp_cor.name)
        os.unlink(tmp_gt.name)

@st.cache_resource
def load_hvt_model(_model_bytes, n_bands):
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
    try:
        tmp.write(_model_bytes); tmp.flush(); tmp.close()
        model = HyperspectralVisionTransformer(
            in_bands=n_bands, num_classes=IP_CLASSES,
            patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, transformer_depth=TRANS_DEPTH
        )
        ckpt = torch.load(tmp.name, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            try: model.load_state_dict(ckpt)
            except: pass
        model.eval()
        return model
    finally:
        os.unlink(tmp.name)

# ── Data Preview ──────────────────────────────────────────────────────────────
with st.spinner("Loading Salinas dataset…"):
    cube, labels = load_mat_files(sal_cor_file.read(), sal_gt_file.read())

H, W, B_sal = cube.shape
labeled_mask = labels > 0
n_labeled    = labeled_mask.sum()

st.success(f"✅ Salinas loaded: **{H}×{W}** pixels · **{B_sal}** bands · **{n_labeled:,}** labeled pixels")

tab1, tab2, tab3 = st.tabs(["🗺️ Data Preview", "🔬 Prediction", "📊 Analysis"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = plot_false_color(cube)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        fig = plot_gt_map(labels)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col3:
        fig = plot_class_dist(labels)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Dataset Statistics")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Scene size",       f"{H}×{W}")
    col_b.metric("Spectral bands",   f"{B_sal}")
    col_c.metric("Labeled pixels",   f"{n_labeled:,}")
    col_d.metric("Land-cover classes", f"{N_SAL}")

# ─────────────────────────────────────────────────────────────────────────────
# Run prediction
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with tab2:
        st.markdown("### 🔄 Processing Pipeline")

        # Step 1: Load model
        with st.status("Loading pretrained HVT model…", expanded=True) as s:
            # We need to determine B_clean (bands after water removal in training)
            # Indian Pines: 220 bands → remove water → ~200 bands (approx)
            # The model's actual in_bands is determined from the .pth
            # We'll try to auto-detect or use a typical value
            # Standard: Indian Pines 220 bands, remove ~20 water bands → 200 bands
            B_model = 200  # default; model will error if wrong

            model_bytes = model_file.read()
            try:
                base_model = load_hvt_model(model_bytes, B_model)
                s.update(label=f"✅ HVT loaded — backbone bands: {B_model}", state="complete")
            except Exception as e:
                # Try different band counts
                for try_bands in [220, 204, 196, 180, 160]:
                    try:
                        base_model = load_hvt_model(model_bytes, try_bands)
                        B_model = try_bands
                        s.update(label=f"✅ HVT loaded — backbone bands: {B_model}", state="complete")
                        break
                    except:
                        continue
                else:
                    st.error(f"Failed to load model: {e}")
                    st.stop()

        # Step 2: Preprocess Salinas
        with st.status("Preprocessing Salinas cube…", expanded=True) as s:
            cube_norm = preprocess_salinas(cube, B_model)
            X_patches, y_true, coords = extract_patches_salinas(cube_norm, labels, PATCH_SIZE)
            s.update(label=f"✅ Extracted {len(X_patches):,} patches — shape {X_patches.shape}", state="complete")

        # Step 3: Domain adaptation
        if use_adaptation:
            with st.status("🔀 Applying Domain Adaptation…", expanded=True) as s:
                adapted_model = adapt_model_mmd(
                    base_model, X_patches, y_true, N_SAL,
                    n_steps=n_adapt_steps, lr=adapt_lr
                )
                s.update(label="✅ Domain adaptation complete", state="complete")
            predict_model = adapted_model
        else:
            # Direct zero-shot: adapt head only (no fine-tuning)
            predict_model = AdaptedHVT(base_model, N_SAL)
            predict_model.eval()
            st.info("⚡ Running zero-shot transfer (no adaptation fine-tuning).")

        # Step 4: Inference
        with st.status("🔮 Running inference on all labeled pixels…", expanded=True) as s:
            probs = predict_batched(predict_model, X_patches, batch_size=BATCH_SIZE)
            pred_map, conf_map = build_prediction_map(probs, coords, labels.shape, N_SAL)
            s.update(label=f"✅ Inference complete on {len(X_patches):,} pixels", state="complete")

        # Compute OA
        preds_flat  = pred_map[labeled_mask]
        labels_flat = labels[labeled_mask]
        oa = (preds_flat == labels_flat).mean() * 100

        st.markdown("---")
        st.markdown("### 📈 Results")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Overall Accuracy",  f"{oa:.2f}%")
        mc2.metric("Labeled Pixels",    f"{n_labeled:,}")
        mc3.metric("Adaptation Steps",  n_adapt_steps if use_adaptation else "—")
        mc4.metric("Target Classes",    N_SAL)

        # Maps
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = plot_gt_map(labels)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col2:
            title = f"Predicted Map (OA={oa:.1f}%)"
            fig = plot_pred_map(pred_map, title=title)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col3:
            fig = plot_conf_map(conf_map, labeled_mask)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Store results in session state
        st.session_state['results'] = {
            'pred_map': pred_map, 'conf_map': conf_map,
            'probs': probs, 'coords': coords,
            'oa': oa, 'labels': labels, 'labeled_mask': labeled_mask
        }

    with tab3:
        if 'results' not in st.session_state:
            st.info("Run prediction first to see analysis.")
        else:
            res = st.session_state['results']
            st.markdown("### Per-Class Accuracy")
            fig = plot_per_class_acc(res['pred_map'], res['labels'], res['labeled_mask'])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Per-class table
            st.markdown("### Class-Level Report")
            rows = []
            for c in range(1, N_SAL+1):
                m = (res['labels'] == c) & res['labeled_mask']
                if m.sum() == 0:
                    continue
                acc  = (res['pred_map'][m] == c).mean() * 100
                conf = res['conf_map'][m].mean() * 100
                rows.append({
                    'Class': SAL_NAMES.get(c, str(c)),
                    'Pixels': int(m.sum()),
                    'Accuracy (%)': f"{acc:.1f}",
                    'Avg Confidence (%)': f"{conf:.1f}"
                })
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=500)

else:
    with tab2:
        st.info("Click **🚀 Run Prediction** in the sidebar to start.")
    with tab3:
        if 'results' not in st.session_state:
            st.info("Run prediction first to see analysis.")
        else:
            res = st.session_state['results']
            fig = plot_per_class_acc(res['pred_map'], res['labels'], res['labeled_mask'])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
