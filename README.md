# 🌾 GreenAid — Multimodal Crop Disease Diagnosis

**Kaggle Competition | Wheat Disease Classification**  https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026
Classify wheat patches into **Health / Rust / Other** using three sensor modalities: RGB imagery, 5-band Multispectral (MS), and 125-band Hyperspectral (HS).

---

## 📁 Repository Structure

```
GreenAid/
├── model.ipynb          # Main notebook: EDA → model → training → submission
├── data/
│   ├── train/
│   │   ├── MS/          # 5-band multispectral .tif files  (Health_N / Rust_N / Other_N)
│   │   ├── HS/          # 125-band hyperspectral .tif files
│   │   └── RGB/         # RGB .png patches (optional)
│   └── val/
│       ├── MS/          # Unlabelled validation set
│       ├── HS/
│       └── RGB/
└── .gitignore
```

---

## 🧠 Approach

### Model — `SimpleFusionNet`

Three independent CNN encoders, each producing a **128-d feature vector**, concatenated and passed through an MLP classifier.

| Encoder | Backbone | Input |
|---|---|---|
| `RGBEncoder` | ResNet-18 (pretrained) | 3 × H × W |
| `MSEncoder` | 2-layer CNN | 5 × H × W |
| `HSEncoder` | 2-layer CNN | 101 × H × W (bands 10–110) |

```
RGB ──► RGBEncoder ──►┐
MS  ──► MSEncoder  ──►├──► concat(384) ──► Linear(256) ──► ReLU ──► Dropout ──► Linear(3)
HS  ──► HSEncoder  ──►┘
```

### Key Design Choices

- **Hyperspectral**: drop noisy end-bands → use bands 10–110 (101 bands)
- **Class-weighted loss**: handles class imbalance automatically
- **Data augmentation**: random 90° rotations + color jitter on RGB
- **Pretrained backbone**: ImageNet ResNet-18 for RGB to maximise transfer learning

---

## 📊 EDA & Visualisations

The notebook generates the following plots:

| Plot | Description |
|---|---|
| `class_distribution.png` | Sample count per class (MS & HS) |
| `sample_vis.png` | MS false-color, HS band-50, and RGB patches per class |
| `spectral_signatures.png` | Mean ± std reflectance curves across wavelengths |
| `vegetation_indices.png` | NDVI / NDRE / GNDVI maps per class |
| `training_curves.png` | Loss & accuracy over epochs |
| `confusion_matrix.png` | Validation confusion matrix |

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 64 × 64 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Epochs | 15 |
| Optimizer | Adam |
| Scheduler | StepLR (step=5, γ=0.5) |
| Train/Val split | 80 / 20 |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install rasterio scikit-learn tqdm matplotlib seaborn pandas numpy Pillow torch torchvision

# 2. Place data in data/train/ and data/val/ as shown above

# 3. Open and run the notebook
jupyter notebook model.ipynb
```

The notebook will train the model, evaluate on the internal validation split, and produce **`submission.csv`** ready for Kaggle upload.

---

## 📤 Output

- **`best_model.pt`** — best checkpoint (highest val accuracy)
- **`submission.csv`** — `Id, Category` predictions for the validation set

---

## 🌿 Vegetation Indices (from MS bands)

| Index | Formula | Detects |
|---|---|---|
| NDVI | (NIR − R) / (NIR + R) | Overall vegetation health |
| NDRE | (NIR − RE) / (NIR + RE) | Chlorophyll / early stress |
| GNDVI | (NIR − G) / (NIR + G) | Canopy density |

Band order in MS files: **Blue(0) · Green(1) · Red(2) · RedEdge(3) · NIR(4)**
