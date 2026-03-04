# 🌾 Beyond Visible Spectrum – AI for Agriculture 2026

> **Competition:** [Kaggle – Beyond Visible Spectrum: AI for Agriculture 2026](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026/overview)

---

## Problem Statement

Classify UAV-captured wheat patches into **3 categories**:

| Class | Meaning |
|-------|---------|
| `Health` | Healthy wheat |
| `Rust` | Wheat rust infection |
| `Other` | Background / other conditions |

Three aligned image modalities are provided per patch:

| Modality | Format | Channels | Spectral Range |
|----------|--------|----------|----------------|
| **RGB** | `.png` | 3 | Visible (450–700 nm) |
| **Multispectral (MS)** | `.tif` | 5 (B/G/R/RE/NIR) | 450–950 nm |
| **Hyperspectral (HS)** | `.tif` | 125 (~4 nm steps) | 450–950 nm |

Data collected via DJI M600 Pro UAV at **60 m altitude** (~4 cm/px), May 2019.

---

## Dataset Structure

```
data/
├── train/
│   ├── RGB/   {Health|Rust|Other}_{N}.png
│   ├── MS/    {Health|Rust|Other}_multi_{N}.tif   (5-band)
│   └── HS/    {Health|Rust|Other}_hyper_{N}.tif   (125-band)
└── val/
    ├── RGB/   val_{hash}.png
    ├── MS/    val_{hash}.tif
    └── HS/    val_{hash}.tif
```

**Train:** 200 samples × 3 classes = **600 per modality**  
**Val:** 300 randomised (unlabelled) samples per modality

---

## Approach

- **Model:** `MultimodalCropNet` — three separate CNN encoders (one per modality) fused via cross-modal `MultiheadAttention`
- **HS pre-processing:** Drop noisy end-bands (keep bands 10–111 → 101 bands)
- **Normalisation:** Per-channel min-max per patch
- **Loss:** `CrossEntropyLoss` with class weights + label smoothing (0.1)
- **LR schedule:** `CosineAnnealingWarmRestarts`
- **Inference:** Standard + Test-Time Augmentation (TTA, 4 rotations)

---

## Quick-Start Guide

### 1 · Install dependencies
```bash
pip install torch torchvision rasterio scikit-learn timm einops \
            tqdm matplotlib seaborn pandas Pillow
```

### 2 · Open the notebook
```bash
jupyter notebook multimodal_crop_disease.ipynb
```

### 3 · Run all cells top-to-bottom

| Cell range | Action |
|------------|--------|
| 1–4 | Setup & EDA |
| 5–7 | Visualise samples & spectral signatures |
| 8–9 | Build `Dataset` classes |
| 10 | Build model |
| 11–13 | Train (GPU recommended) |
| 14–15 | Plot curves & confusion matrix |
| 16 | Generate **`submission.csv`** |
| 17 | Generate **`submission_tta.csv`** (TTA, better accuracy) |
| 18 | Vegetation index visualisations (NDVI / NDRE / GNDVI) |

> **Tip:** Submit `submission_tta.csv` for best leaderboard score.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Model, training, data transforms |
| `rasterio` | Read multi-band `.tif` GeoTIFF files |
| `numpy` / `pandas` | Numerical ops & CSV handling |
| `scikit-learn` | Metrics (classification report, confusion matrix) |
| `Pillow` | RGB `.png` loading |
| `matplotlib` / `seaborn` | Plots & heatmaps |
| `tqdm` | Progress bars |
| `einops` | Tensor reshaping (optional extensions) |
| `timm` | Pre-trained backbone zoo (available for future upgrades) |

---

## Submission Format

```csv
Id,Category
val_000a83c1.tif,Health
val_00a704b1.tif,Rust
...
```
