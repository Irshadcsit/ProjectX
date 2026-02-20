## Medical Imaging Classification, Visual Language Models

**Author:** Irshad Khan | 

[![Open Data Analysis in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/01_data_analysis.ipynb)
[![Open Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/02_train.ipynb)
[![Open Evaluation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/03_evaluate.ipynb)

---

## Repository Structure

```
repository/
├── data/
│   ├── __init__.py
│   └── dataset.py              ← PneumoniaMNIST pipeline (transforms, loaders, class weights)
├── models/
│   ├── __init__.py
│   ├── registry.py             ← Model registry — swap models with one line
│   ├── efficientnet.py         ← EfficientNet-B0 (recommended)
│   ├── resnet.py               ← ResNet-18 (baseline)
│   └── custom_cnn.py           ← Custom CNN from scratch (lower bound)
├── task1_classification/
│   ├── train.py                ← CLI training script
│   ├── evaluate.py             ← CLI evaluation script
│   └── compare_models.py       ← Side-by-side model comparison
├── task2_report_generation/    ← [Task 2]
├── task3_retrieval/            ← [Task 3]
├── notebooks/
│   ├── 01_data_analysis.ipynb  ← EDA — run first
│   ├── 02_train.ipynb          ← Training — change MODEL_NAME to swap model
│   └── 03_evaluate.ipynb       ← Evaluation + comparison — run last
├── reports/
│   ├── task1_classification_report.md
│   └── outputs/                ← Generated plots
│       ├── efficientnet/       ← Plots for EfficientNet-B0
│       ├── resnet/             ← Plots for ResNet-18
│       ├── custom_cnn/         ← Plots for Custom CNN
│       └── model_comparison.png
├── requirements.txt
└── README.md
```

---

## Task 1: CNN Classification

### Quick Start — Google Colab (Recommended)

1. Open `notebooks/01_data_analysis.ipynb` → explore the dataset
2. Open `notebooks/02_train.ipynb` → set `MODEL_NAME`, train
3. Open `notebooks/03_evaluate.ipynb` → evaluate + compare all models

Each notebook is **self-contained** — it clones the repo and imports `.py` modules automatically.

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### Swapping Models

Change one variable in any notebook or script:

```python
# notebooks/02_train.ipynb  ─  Cell 3
MODEL_NAME = 'efficientnet'   # ← change to 'resnet' or 'custom_cnn'
```

Or via CLI:

```bash
cd task1_classification
python train.py    --model efficientnet --epochs 20
python train.py    --model resnet       --epochs 20
python train.py    --model custom_cnn  --epochs 30 --lr 3e-4

python evaluate.py --model efficientnet
python evaluate.py --model resnet
python compare_models.py                            # compare all
```

### Output Structure (per model)

```
outputs/
├── efficientnet/
│   ├── best_model.pth
│   ├── training_history.json
│   ├── test_metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── failure_cases.png
│   ├── confidence_distribution.png
│   └── sample_predictions.png
├── resnet/           ← same structure
├── custom_cnn/       ← same structure
└── model_comparison.png
```

### Adding a New Model

1. Create `models/your_model.py` with a class that has `forward(x)` and `unfreeze_all()` methods
2. Add it to `models/registry.py`:

```python
from .your_model import YourModel

MODEL_REGISTRY = {
    "efficientnet": EfficientNetB0Classifier,
    "resnet":       ResNet18Classifier,
    "custom_cnn":   CustomCNNClassifier,
    "your_model":   YourModel,          # ← add here
}
```

3. Use it: `python train.py --model your_model`

### Expected Results

| Model | Accuracy | AUC | Recall | Training Time (T4) |
|-------|----------|-----|--------|--------------------|
| EfficientNet-B0 | ~0.90–0.93 | ~0.95–0.97 | ~0.92–0.96 | ~15–20 min |
| ResNet-18 | ~0.87–0.91 | ~0.93–0.96 | ~0.90–0.94 | ~12–18 min |
| Custom CNN | ~0.82–0.87 | ~0.88–0.93 | ~0.85–0.90 | ~10–15 min |

---

## References

- Yang et al., "MedMNIST v2," Scientific Data, 2023.
- Tan & Le, "EfficientNet," ICML 2019.
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
