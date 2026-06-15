# Word Embeddings for Khmer Text

A comprehensive implementation of word embeddings for Khmer (Cambodian) text using the Skip-gram model with negative sampling, PCA visualization, and neural language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Key Findings](#key-findings)
- [References](#references)

## Overview

This project implements a complete word embeddings pipeline for Khmer text as part of an NLP mini project. It covers:

| Part | Description |
|------|-------------|
| **Part I** | Skip-gram model with negative sampling (50D embeddings) |
| **Part II** | PCA visualization of word embeddings |
| **Part III** | Neural language model with fixed embeddings from Part I |
| **Part IV** | Learning embeddings from scratch + comparison with Skip-gram |

## Technologies Used

- **PyTorch** — Deep learning framework for building neural models
- **khmer-nltk** — Khmer word tokenization
- **scikit-learn** — PCA dimensionality reduction
- **NumPy / Pandas** — Data processing
- **Matplotlib** — Visualization
- **Streamlit** — Interactive web application (`app.py`)

## Dataset

The corpus is sourced from three Khmer Wikipedia articles about Cambodian temples:

- **File:** `temples.txt`
- **Content:** Articles about Angkor Wat, Banteay Srei, and Siem Reap province
- **Size:** 45,920 characters
- **After tokenization:** 11,521 tokens
- **Vocabulary (after filtering):** 182 unique words (frequency ≥ 10)

## Project Structure

```
├── main.ipynb            # Main Jupyter notebook with full implementation
├── app.py                # Streamlit web application
├── temples.txt           # Khmer text corpus
├── requirements.txt      # Python dependencies
├── project_documentation_kh.md  # Khmer-language documentation
├── pca_skipgram.png      # PCA visualization of Skip-gram embeddings
├── pca_comparison.png    # PCA comparison (Skip-gram vs Scratch)
├── hyperparameter_tuning.png     # Hyperparameter tuning results
├── mini_project_3_report.pdf     # Final report (PDF)
└── mini_project_3.pdf    # Project brief
```

## Installation

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify khmer-nltk setup (downloads tokenization model on first use)
python -c "from khmernltk import word_tokenize; print('khmer-nltk ready')"
```

**Key packages:**
- `torch` — Neural network models
- `khmer-nltk` — Khmer tokenization
- `scikit-learn` — PCA
- `matplotlib` — Visualization
- `streamlit` — Web app (optional)

## Usage

### Run the Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

The notebook is self-contained and organized into 4 parts:

1. **Part I — Skip-gram Model:** Tokenizes Khmer text, builds vocabulary, generates (center, context) pairs, and trains a Skip-gram model with negative sampling.
2. **Part II — PCA Visualization:** Reduces 50D embeddings to 2D using PCA and plots words in 2D space.
3. **Part III — Neural LM (Fixed):** Trains a neural language model using 5 previous words to predict the next word, with embeddings frozen from Part I.
4. **Part IV — Neural LM (Scratch):** Same architecture as Part III but learns embeddings from random initialization. Compares results.
5. **Hyperparameter Tuning:** Tests different embedding dimensions, window sizes, and negative sample counts.

### Launch the Web App

```bash
streamlit run app.py
```

## Results

### Part I: Skip-gram Model

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 50 |
| Context window | ±4 |
| Negative samples | 2 |
| Vocabulary size | 182 |
| Total parameters | 18,200 |
| Final loss | 1.8832 |

Words closest to **ប្រាសាទ** (temple):
- បន្ទាយស្រី (Banteay Srei): 0.9694
- អង្គរវត្ត (Angkor Wat): 0.9545
- អង្គរតូច (Angkor Thom): 0.9496

### Part II: PCA Visualization

- **PC1 explains:** 38.68% of variance
- **PC2 explains:** 14.17% of variance
- **Total explained:** 52.85%

![PCA Visualization](pca_skipgram.png)

### Part III vs Part IV: Neural Language Models

| Metric | Part III (Fixed) | Part IV (Scratch) |
|--------|:----------------:|:-----------------:|
| **Loss** | 4.7324 | **3.7992** |
| **Perplexity** | 113.55 | **44.62** |
| PCA 2D Variance | 52.85% | 8.57% |

![Comparison](pca_comparison.png)

**Key Insight:** Scratch embeddings achieve **2.5× lower perplexity** than fixed Skip-gram embeddings for the next-word prediction task. However, Skip-gram embeddings capture better semantic relationships for similarity tasks.

## Hyperparameter Tuning

| Dim | Window | Neg Samples | Loss | PCA Variance |
|:---:|:------:|:-----------:|:----:|:-----------:|
| 30 | 2 | 2 | 1.9987 | 55.65% |
| 50 | 4 | 2 | 1.8832 | 52.85% |
| 50 | 6 | 2 | 1.7853 | **71.04%** |
| 80 | 4 | 2 | 1.6844 | 57.04% |
| 100 | 6 | 1 | **1.3296** | 57.79% |

![Hyperparameter Tuning](hyperparameter_tuning.png)

### Recommendations

- **Best for similarity tasks:** dim=50, window=6, neg=2 (highest PCA variance)
- **Best for prediction tasks:** dim=100, window=6, neg=1 (lowest loss)
- **Recommended all-rounder:** dim=50, window=6, neg=2

## Key Findings

1. **Larger context windows** generate more training pairs and improve both loss and PCA variance.
2. **Scratch embeddings** outperform fixed Skip-gram embeddings for next-word prediction (44.62 vs 113.55 perplexity).
3. **Skip-gram embeddings** capture better semantic structure in PCA space (52.85% vs 8.57% explained variance).
4. **Khmer tokenization** with `khmer-nltk` works effectively for word-level NLP tasks.
5. **Negative sampling** with 2 samples provides a good balance between training speed and embedding quality.

## References

- Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." (2013)
- Mikolov, T., et al. "Distributed Representations of Words and Phrases and their Compositionality." (2013)
- [khmer-nltk](https://github.com/viclaw/khmer-nltk) — Khmer NLP toolkit

---

*Created for M1 DAS S2 — NLP Mini Project 3*
