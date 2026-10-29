# Word Sense Disambiguation (WSD) using BERT and Logistic Regression

A Natural Language Processing (NLP) mini-project that identifies the intended meaning of an ambiguous word based on its context. The system uses a **fine-tuned BERT encoder** for contextual representations, **lemma/POS-specific Logistic Regression classifiers** for sense prediction, **WordNet** for sense definitions and fallback predictions, and a **Streamlit** web application for interactive inference.

## Overview

Words can have different meanings depending on the sentence in which they appear. **Word Sense Disambiguation (WSD)** is the task of selecting the correct meaning (sense) of a word from its context.

For example:

- **“He deposited money at the bank.”** → *bank*: a financial institution.
- **“They sat on the bank of the river.”** → *bank*: the land beside a river.

This project predicts a WordNet synset for a selected target word and retrieves its definition.

## Features

- Context-aware word sense prediction using fine-tuned BERT.
- Training and evaluation on the **SemCor** sense-annotated corpus.
- Target-word-aware tokenization and contextual embedding extraction.
- Separate classifiers for each lemma and part-of-speech (POS) combination.
- WordNet-based fallback when the target lemma/POS was not seen during training.
- Evaluation with accuracy, precision, recall, F1-scores, and semantic confusion analysis.
- Interactive interface developed with **Streamlit**.
- Saved model artifacts for inference without retraining, once generated locally.
- Trained artifacts are intentionally excluded from GitHub to keep the repository lightweight.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | BERT fine-tuning and inference |
| Hugging Face Transformers | BERT model and tokenizer |
| NLTK / SemCor | Annotated training corpus |
| NLTK / WordNet | Synsets, sense definitions, and fallback |
| Scikit-learn | StandardScaler, Logistic Regression, and evaluation |
| NumPy / Pandas | Data processing |
| Joblib | Saving trained classifiers |
| Streamlit | Web interface |
| Kaggle Notebooks | GPU-backed model training |

## Model Architecture

```text
Input sentence + target word / position
                |
                v
    Text normalization and token alignment
                |
                v
      Fine-tuned BERT (bert-base-uncased)
                |
                v
 Target WordPiece contextual embeddings
 (mean of last four layers and target pieces)
                |
                v
          StandardScaler
                |
                v
    Lemma/POS-specific Logistic Regression
                |
                v
         Predicted WordNet synset
                |
                v
       WordNet sense definition
```

For lemma/POS combinations without a trained classifier, the pipeline falls back to the first matching WordNet synset. Groups with only one observed training sense use a constant classifier.

## Dataset and Training Pipeline

**Dataset:** [SemCor](https://www.nltk.org/howto/corpus.html), accessed through NLTK, with semantic labels linked to WordNet synsets.

The notebook is organized into five parts:

1. **Dataset acquisition:** Load annotated SemCor sentences, target words, lemma/POS information, and WordNet sense labels.
2. **Preprocessing:** Normalize text while preserving important context, word order, punctuation, negation, and target alignment. Split sentence groups into training, validation, and test partitions (approximately 64% / 16% / 20%).
3. **Contextual representation:** Fine-tune `google-bert/bert-base-uncased` with a temporary sense classification head and extract target-specific contextual vectors using the last four encoder layers.
4. **Word sense classification:** Fit a `StandardScaler` and train separate Logistic Regression models by lemma/POS, using `C = 0.03` in the saved run.
5. **Evaluation:** Evaluate held-out predictions, compare against a Most Frequent Sense (MFS) baseline, and analyze errors and sense confusions.

**Important:** The BERT model is fine-tuned; the final Logistic Regression models operate on the resulting BERT embeddings. This is not an end-to-end BERT classification head at inference time.

## Evaluation Results

Results recorded in the included training notebook:

| Metric | Result |
|---|---:|
| Test examples | 45,583 |
| **Test accuracy** | **80.60%** |
| Validation accuracy (final pipeline) | 80.45% |
| Macro F1-score | 70.54% |
| Weighted F1-score | 78.69% |
| Most Frequent Sense baseline accuracy | 71.88% |
| Missing predictions | 0 |

The final system improves on the MFS baseline by approximately **8.73 percentage points** on this test split.

### Accuracy by part of speech

| POS | Accuracy |
|---|---:|
| Adjectives | 86.51% |
| Nouns | 84.29% |
| Adverbs | 90.30% |
| Verbs | 72.71% |

Verbs and rare or unseen senses remain more difficult to classify. Results depend on the exact dataset split and saved training configuration; they should not be interpreted as a benchmark comparison against other WSD studies with different evaluation protocols.

## Project Structure

The repository contains the application, notebook, and documentation. **The trained `artifacts/` directory is local-only and is ignored by Git.**


```text
WSD-Stremlit/
├── .gitignore           # Excludes artifacts/ and venv/
├── artifacts/           # LOCAL ONLY — not included in GitHub
│   ├── encoder/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── tokenizer/
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── run_config.json
│   ├── sense_mappings.json
│   └── wsd_model.joblib
├── notebook/
│   └── nlp-mini-project-wsd.ipynb
├── app.py
├── requirements.txt
└── README.md
```

**Note:** The `artifacts/` directory in this tree represents the expected *local* setup. It is not available after cloning the repository; you must restore or regenerate it before running the app.

The training notebook also generates `data/` and `evaluation/` folders in its Kaggle output directory. These are not required to be present in the Streamlit project folder for inference.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/ZAKARIAGHAZI/WSD-Stremlit.git
cd WSD-Stremlit
```


### 2. Create a virtual environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the application requires NLTK WordNet resources, download them once:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Prepare the trained model artifacts (required)

**Model weights and trained classifiers are not hosted in this GitHub repository.** The `artifacts/` folder is excluded by the root `.gitignore` because it contains large generated files. Cloning the repository and installing the dependencies alone is **not enough** to run inference.

Choose one of these approaches:

1. **Already trained the model?** Copy your existing `artifacts/` directory from your own trusted backup or Kaggle training output into the repository root, next to `app.py`.
2. **Need to regenerate the model?** Run `notebook/nlp-mini-project-wsd.ipynb` in Kaggle with a GPU, then download the generated `artifacts/` directory and place it in the repository root (see [Reproducing the Training](#reproducing-the-training)).

The required files include `encoder/model.safetensors`, the encoder configuration, tokenizer files, `run_config.json`, `sense_mappings.json`, and `wsd_model.joblib`. All files must come from the **same training run**.

Do **not** commit the generated artifacts. Your root `.gitignore` should contain `artifacts/` and `venv/`.

> **Security note:** Only load `wsd_model.joblib` if it comes from a trusted source. Joblib deserialization can execute arbitrary code.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit (usually `http://localhost:8501`). Use the interface to provide a sentence and select or specify the target word according to the fields available in the app.

## Reproducing the Training

The notebook is located at `notebook/nlp-mini-project-wsd.ipynb` and was developed for a GPU-enabled Kaggle environment. To regenerate the trained model:

1. Open the notebook in Kaggle and enable GPU acceleration.
2. Run the cells in order, including SemCor/WordNet download, preprocessing, BERT fine-tuning, classifier training, and evaluation.
3. Retrieve the generated `artifacts/` directory from `/kaggle/working/wsd_simple/` (or the configured notebook output directory).
4. Place that directory next to `app.py` to use the saved model in Streamlit.

The notebook also writes datasets and embeddings under `data/`, and evaluation reports, confusion tables, and figures under `evaluation/`.

## Limitations and Future Improvements

- Distinguishing closely related meanings remains difficult, especially for verbs.
- Rare and unseen senses have lower accuracy than frequent senses.
- The classifier relies on the coverage and definitions of WordNet.
- Running BERT inference can be slow on CPU-only systems.
- Potential improvements include gloss-aware WSD approaches such as GlossBERT, richer candidate-ranking methods, and further systematic evaluation.

## References

- [SemCor and NLTK](https://www.nltk.org/)
- [WordNet — Princeton University](https://wordnet.princeton.edu/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Project type:** NLP mini-project — Word Sense Disambiguation (WSD).
