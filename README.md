# Emotion Detection from Text: CNN vs DistilBERT vs RoBERTa

Comparative study benchmarking three NLP architectures on a 6-class tweet emotion classification task.

## Results

| Model | Test Accuracy | Epochs |
|---|---|---|
| CNN (Conv1D) | 63.52% | 5 |
| DistilBERT | 78.18% | 19 |
| **RoBERTa** | **80.90%** | 15 |

## Dataset

**Source:** Kaggle Emotion Dataset — 34,792 labeled text samples.

**Target classes (6):** joy, sadness, fear, anger, surprise, neutral

Two low-frequency classes (shame, disgust) were dropped to reduce noise. The remaining data was augmented via synonym replacement and random deletion to balance minority classes, bringing the final training set to 37,140 samples.

| Emotion | Samples (final) |
|---|---|
| joy | 10,615 |
| sadness | 6,411 |
| fear | 5,114 |
| anger | 5,000 |
| neutral | 5,000 |
| surprise | 5,000 |

## Models

### CNN (Conv1D)

A 1D convolutional network built as a fast baseline. Text goes through custom preprocessing: emotion-aware stopword filtering (preserving negations, intensifiers, contrast words), Porter stemming, and GloVe-style embedding layer.

Architecture: `Embedding(10K vocab, 64d) → Conv1D(128 filters, k=5) → GlobalMaxPool → Dense(128) → Dropout → Dense(6)`

Trained for 5 epochs before early stopping. Reaches 63.52% — strong on high-signal emotions (joy, sadness, fear), struggles with neutral vs. surprise.

### DistilBERT

Fine-tuned `distilbert-base-uncased` (66M parameters) with a classification head on top. No stemming — WordPiece tokenizer preserves morphological structure. Augmented data only; non-English samples filtered out.

Trained for 19 epochs (early stop at patience=3). Reaches 78.18% on test set, with neutral scoring 0.92 F1 — the clearest improvement over CNN.

### RoBERTa

Fine-tuned `roberta-base` with AdamW optimizer, linear learning rate scheduler, and 10% warmup steps. Byte-Pair Encoding handles subword units without losing semantic context. Same augmentation pipeline as DistilBERT.

Trained for 15 epochs. Reaches 80.90% — highest accuracy across all three models, consistent precision and recall across all 6 classes.

## Notebooks

| Notebook | Description |
|---|---|
| `emotion_cnn.ipynb` | Full CNN pipeline: preprocessing, training, evaluation |
| `distilbert_emotion.ipynb` | DistilBERT fine-tuning and evaluation |
| `roberta_emotion.ipynb` | RoBERTa fine-tuning and evaluation |
| `multi_model_comparison.ipynb` | Side-by-side metrics, confusion matrices, case studies |

## Key Findings

RoBERTa's bidirectional contextual encoding handles ambiguous, short tweet text better than the other two. DistilBERT gives 95% of RoBERTa's accuracy at a fraction of the compute cost — the practical choice for real-time deployment. The CNN remains a solid starting point: fast to train, interpretable, and useful for prototyping or lightweight deployment where a 63% baseline is acceptable.

## Stack

Python · PyTorch · HuggingFace Transformers · scikit-learn · NLTK · Pandas · Matplotlib

## Team

| Name | 
|---|---|
| Marcel Ngoyi |
| Imo Christabel | 
| Romina Sonboldoust | 
| Noura Hussien |

