## Medical Imaging Classification, Visual Language Models

**Author:** Irshad Khan  

Main Notebook:
For a quick review, you can open the main notebook directly in Colab.
For a comprehensive evaluation, I recommend running each notebook individually: 
Task 1 can be executed either in Colab or on a local machine, 
while Task 2 requires GPU resources, so Colab is the preferred option.
[![Open Main Notebook in Colab]](https://colab.research.google.com/github/Irshadcsit/ProjectX/blob/main/Main_notebook.ipynb)

### Tas1:
Check Individual:
[![Open Data Analysis in Colab]](https://colab.research.google.com/github/Irshadcsit/ProjectX/blob/main/01_data_analysis.ipynb)

[![Open Classifier Training in Colab]](https://colab.research.google.com/github/Irshadcsit/ProjectX/blob/main/01_train_classifier.ipynb)

[![Open Classifier Evaluation in Colab]](https://colab.research.google.com/github/Irshadcsit/ProjectX/blob/main/01_evaluate_classifier.ipynb)

### Task2:
[![Open Report Generation using VLLM in Colab]](https://colab.research.google.com/github/Irshadcsit/ProjectX/blob/main/task2_colab_new.ipynb)
---

## Repository Structure
#Folders:
Classification: 


## Task 1: CNN Classification

### Quick Start — Google Colab (Recommended)
1. Open `01_data_analysis.ipynb` → explore the dataset and analyze the results are available in data_analysis folder
2. Open `02_train.ipynb` → set `MODEL_NAME`, train
3. Open `03_evaluate.ipynb` → evaluate each model, + compare all models

Each notebook is **self-contained** — it clones the repo and imports `.py` modules automatically.

### Local Setup

```bash
git clone 'https://github.com/Irshadcsit/ProjectX.git'
cd ProjectX
pip install -r requirements.txt
```

### Swapping Models

Change one variable in any notebook or script:

```python
# 01_train_classifier.ipynb  ─  Cell 3
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
├── Model_Name(e.g., efficientnet)/
│   ├── best_model.pth               # missing for vit due to large size.
│   ├── training_history.json
│   ├── test_metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── failure_cases.png
│   ├── confidence_distribution.png
│   └── sample_predictions.png
├── resnet/           ← same structure
└── model_comparison.png
└── auc_comparison.png    
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

### Task 2: Visual Language Models (Report Generation)
To run Task 2 in Colab:
You need a Hugging Face access token to use gated models such as MedGemma.
Generate a token from your Hugging Face account settings.
Store the token securely in Colab Secrets to avoid exposing it in plain text.
A script for token registration is already included in the notebook, so authentication is handled automatically during runtime.

### Results
See Task1 and Task2 reports

---

## Dataset Reference
- Yang et al., "MedMNIST v2," Scientific Data, 2023.
