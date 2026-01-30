# Character-Level Orthographic Correction with Transformers

This repository contains the implementation of a **Transformer-based character-level orthographic corrector**, developed as part of the *Hands-on Generative AI* course at the Technical University of Munich.

The project formulates spelling correction as a **sequence-to-sequence learning task**, where a noisy input sentence is mapped to its corrected version at the character level. The model is trained on synthetically noisified English Wikipedia text and evaluated using standard error-based metrics.

A complete description of the model architecture, dataset construction, training procedure, and evaluation methodology is provided in the accompanying **project report**.

---

## Repository Overview


```text
orthocorrector/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts
│   ├── evaluate.py
│   └── train.py
└── src
    ├── data
    │   ├── wiki_sentences_10MB.txt
    │   ├── wiki_sentences_20MB.txt
    │   ├── wiki_sentences_30MB.txt
    │   ├── wiki_sentences_40MB.txt
    │   └── wiki_sentences_50MB.txt
    ├── models
    │   ├── checkpoints
    │   └── final
    ├── orthocorrector
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── data.py
    │   ├── evaluation.py
    │   ├── io.py
    │   ├── model.py
    │   └── training.py
    ├── orthocorrector.egg-info
    └── results
        ├── Summary_table.csv
        ├── test_predictions_10MB.csv
        ├── test_predictions_20MB.csv
        ├── test_predictions_30MB.csv
        └── test_predictions_40MB.csv
```

The repository contains two main folders: `src`, `scripts`.
The `src` folder stores:
* the **orthocorrector** folder, which includes all core components of the project(`data.py`, `model.py`, `training.py`, `evaluation.py`, `io.py`)
* the **data** folder, containing the datasets used in training and evaluation, 
* the **models** folder, containing trained models (stored in the **final** subfolder)
* the **results** folder, containing the evaluation outputs and metrics

The `scripts` folder contains the executable entry points:
* **`train.py`** : runs the training process and saves checkpoints and final models
* **`evaluate.py`** : runs the evaluation process and saves the resulting metrics

---

## Dataset

The main dataset characteristics are:
- Source: [English Wikipedia (plaintext)](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english)
- Character set: ASCII only
- Synthetic character-level noise simulating typographical errors
- Training subsets:
  - 10MB
  - 20MB
  - 30MB
  - 40MB
- A shared test set is derived from the 50MB subset

Details on preprocessing, sentence splitting, and noisification are described in the report (Section 3).

---

## Model and Evaluation

The main model characteristics are:
- Architecture: Encoder–Decoder Transformer
- Tokenization: Character-level
- Evaluation metrics:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Additional qualitative indicators (Exact, Improved, Regressed, etc.)

All experimental details and results are documented in the report.

---

## Setup Instructions

#### Create and activate a conda environment

This step is optional but recommended.

```bash
conda create -n orthocorrector python=3.10 -y
conda activate orthocorrector
```

#### Install dependencies

All required libraries are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

#### Install the project in editable mode

This allows the project modules to be imported across scripts.

```bash
pip install -e .
```

---

## Running the Code


#### Train models

```bash
python scripts/train.py
```

This trains Transformer models on the 10MB, 20MB, 30MB, and 40MB datasets and saves checkpoints and final models under `models/`.




#### Evaluate models
```bash
python scripts/evaluate.py
```

This evaluates all trained models on the shared test set and writes evaluation results to disk.

---

## Report

For a detailed explanation of the methodology, architecture, training procedure, and results, please refer to the **project report**:

**Character-Level Orthographic Correction with Transformers**
*Simon Coxner, Giacomo Cordella*
Technical University of Munich
