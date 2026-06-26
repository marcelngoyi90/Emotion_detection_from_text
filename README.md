# Emotion Detection from Text: CNN vs DistilBERT vs RoBERTa

A comparative NLP project benchmarking three model families on a 6-class tweet emotion classification task: a CNN baseline, DistilBERT, and RoBERTa.

## Results

| Model | Test Accuracy | Epochs |
|---|---:|---:|
| CNN (Conv1D) | 63.52% | 5 |
| DistilBERT | 78.18% | 19 |
| **RoBERTa** | **80.90%** | 15 |

RoBERTa achieved the strongest overall performance, while DistilBERT delivered a strong accuracy-to-compute tradeoff. The CNN remains useful as a fast baseline for prototyping and lightweight deployment.

## Dataset

Source: Kaggle Emotion Dataset with 34,792 labeled text samples.

Target classes:

- Joy
- Sadness
- Fear
- Anger
- Surprise
- Neutral

Low-frequency `shame` and `disgust` classes were removed to reduce label noise. The remaining training data was augmented using synonym replacement and random deletion, producing a final training set of 37,140 samples.

| Emotion | Final Samples |
|---|---:|
| Joy | 10,615 |
| Sadness | 6,411 |
| Fear | 5,114 |
| Anger | 5,000 |
| Neutral | 5,000 |
| Surprise | 5,000 |

## Model Approaches

### CNN Baseline

A 1D convolutional model using custom preprocessing, emotion-aware stopword filtering, stemming, and an embedding layer.

Architecture:

```text
Embedding -> Conv1D -> GlobalMaxPooling -> Dense -> Dropout -> Dense(6)
```

### DistilBERT

Fine-tuned `distilbert-base-uncased` with a classification head. DistilBERT improved performance substantially while keeping the model lighter than full BERT-style architectures.

### RoBERTa

Fine-tuned `roberta-base` with AdamW, a linear learning-rate scheduler, and warmup steps. RoBERTa achieved the highest test accuracy and the most consistent class-level performance.

## Notebooks

| Notebook | Description |
|---|---|
| `emotion_cnn.ipynb` | CNN preprocessing, training, and evaluation |
| `distilbert_emotion.ipynb` | DistilBERT fine-tuning and evaluation |
| `roberta_emotion.ipynb` | RoBERTa fine-tuning and evaluation |
| `multi_model_comparison.ipynb` | Side-by-side metrics, confusion matrices, and error analysis |

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NLTK
- Pandas
- Matplotlib

## Key Takeaways

- Transformer models handled short and ambiguous tweet text better than the CNN baseline.
- DistilBERT is a practical option when inference speed matters.
- RoBERTa achieved the best accuracy and strongest class-level balance.
- Data cleaning, class balancing, and augmentation had a meaningful impact on model quality.

## Team

- Marcel Ngoyi
- Imo Christabel
- Romina Sonboldoust
- Noura Hussien
