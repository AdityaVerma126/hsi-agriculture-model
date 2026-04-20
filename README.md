# 🌿 HSI Salinas Predictor — Streamlit App

Hyperspectral Image Classification on the **Salinas-A** dataset using a pretrained
**Hyperspectral Vision Transformer (HVT)** with **MMD domain adaptation** from Indian Pines.

---

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Files Needed

| File | Description |
|---|---|
| `best_hvt_model.pth` | Pretrained HVT trained on Indian Pines |
| `salinas_corrected.mat` | Salinas 204-band corrected hyperspectral cube |
| `salinas_gt.mat` | Salinas ground-truth label map |
| `salinas.mat` | (Optional) Salinas raw 224-band cube |

Upload all files directly in the sidebar of the running app.

---

## Model Architecture

```
Input (N, B, P, P) patches
        │
        ├──▶ SpectralCNN3D  ──▶ (N, 32, P, P)   [3D convolutions on spectral dim]
        │
        ├──▶ SpatialCNN2D   ──▶ (N, 64, P, P)   [Depthwise 2D CNN]
        │
        └──▶ Concatenate    ──▶ (N, 96, P, P)
                │
                ▼
        MorphologicalLayer  ──▶ (N, 96, P, P)   [Differentiable dilation/erosion]
                │
                ▼
        Flatten to tokens   ──▶ (N, P², 96)
                │
                ▼
        Linear projection   ──▶ (N, P², 128)
                │
                ▼
        Transformer Encoder ──▶ (N, 128)         [Multi-head self-attention, CLS token]
                │
                ▼
        MLP Classifier      ──▶ (N, C)           [Adapted for Salinas classes]
```

## Domain Adaptation (MMD)

The pretrained model was trained on Indian Pines (220 bands, 16 land-cover classes).
Salinas-A has different spectral characteristics and 16 vegetation classes.

Adaptation steps:
1. **Spectral interpolation** — Salinas bands are interpolated to match model's expected band count
2. **New classification head** — The output layer is replaced for 16 Salinas classes
3. **Entropy minimization** — Backbone is frozen; new head is fine-tuned on target domain features
   using entropy + diversity loss (no labels required — self-supervised)

## Salinas-A Classes
1. Weeds-1             9. Soil-vineyard  
2. Weeds-2            10. Corn-senesced  
3. Fallow             11. Lettuce-4wk  
4. Fallow-rough-plow  12. Lettuce-5wk  
5. Fallow-smooth      13. Lettuce-6wk  
6. Stubble            14. Lettuce-7wk  
7. Celery             15. Vineyard-untrained  
8. Grapes-untrained   16. Vineyard-vertical  
