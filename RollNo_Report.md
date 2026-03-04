# ML Home Assignment Report
## COVID-19 Fake News Detection
**Name:** ____________________  
**Roll No:** ____________________  
**Date:** March 2, 2026  
**Dataset:** Constraint@AAAI-2021 (COVID-19 Fake News Detection)

---

## 1. Dataset Summary

| Split | Total Samples | Fake (0) | Real (1) |
|-------|--------------|----------|----------|
| Train (80%) | `____` | `____` | `____` |
| Val   (10%) | `____` | `____` | `____` |
| Test  (10%) | `____` | `____` | `____` |
| **Total**   | **10600** | **5055** | **5545** |

---

## 2. Preprocessing Pipeline

| Step | Operation | Reason |
|------|-----------|--------|
| 1 | Lowercase | Normalize text case |
| 2 | Emoji to text (unicode demojize) | Preserve semantic meaning of emojis |
| 3 | Hashtag cleaning (#StayHome to StayHome) | Keep stance-bearing words |
| 4 | Remove URLs | URLs carry no semantic value |
| 5 | Remove @mentions | Not informative for classification |
| 6 | Remove special characters | Reduce noise |
| 7 | Tokenize (NLTK word_tokenize) | Split into tokens |
| 8 | Remove stopwords (keep negations) | Remove noise, preserve not/no/never |
| 9 | Lemmatize (WordNetLemmatizer) | Normalize word forms |
| 10 | Rejoin tokens | Reconstruct clean string |

---

## 3. TF-IDF Vectorization

| Parameter | Value |
|-----------|-------|
| max_features | 20,000 |
| ngram_range | (1, 2) |
| sublinear_tf | True |
| min_df | 2 |
| max_df | 0.95 |
| Fit on | Train only |

**Vocabulary size after fitting:** `____`
**Feature matrix shape (train):** `(_____, 20000)`

---

## 4. Model Results

### 4.1 K-Nearest Neighbor

**Tuning:** GridSearchCV (5-fold CV, scoring=F1)

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| n_neighbors | [3, 5, 7, 11, 15] | `____` |
| metric | [euclidean, manhattan, cosine] | `____` |
| weights | [uniform, distance] | `____` |

**Best CV F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

---

### 4.2 Logistic Regression

**Tuning:** GridSearchCV (5-fold CV, scoring=F1)

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| C | [0.01, 0.1, 1, 10, 100] | `____` |
| penalty | [l1, l2] | `____` |
| solver | [liblinear, saga] | `____` |
| max_iter | [200, 500] | `____` |

**Best CV F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

---

### 4.3 Support Vector Machine (Linear Kernel)

**Tuning:** GridSearchCV (5-fold CV, scoring=F1)

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| C | [0.001, 0.01, 0.1, 1, 10] | `____` |
| loss | [hinge, squared_hinge] | `____` |

**Best CV F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

---

### 4.4 K-Means Clustering

**Tuning:** Manual Grid Search with majority-vote label mapping

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| init | [k-means++, random] | `____` |
| max_iter | [100, 300] | `____` |
| n_init | [10, 20] | `____` |

**Best Train F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

> Note: K-Means is unsupervised. Cluster IDs were mapped to class labels using majority voting.

---

### 4.5 Neural Network (PyTorch)

**Architecture:**
Input (TF-IDF) -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear(1) -> Sigmoid

**Tuning:** Manual Grid Search (best Val F1)

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| hidden_dim | [256, 512] | `____` |
| dropout | [0.2, 0.3, 0.5] | `____` |
| learning_rate | [1e-3, 1e-4] | `____` |
| batch_size | [32, 64, 128] | `____` |
| epochs | [20, 30] | `____` |

**Best Val F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

---

### 4.6 Gradient Boosting (Ensemble)

**Tuning:** RandomizedSearchCV (20 iterations, 5-fold CV, scoring=F1)

| Hyperparameter | Values Searched | Best Value |
|----------------|----------------|------------|
| n_estimators | [50, 100, 200, 300] | `____` |
| learning_rate | [0.01, 0.05, 0.1, 0.2] | `____` |
| max_depth | [3, 4, 5, 7] | `____` |
| subsample | [0.7, 0.8, 0.9, 1.0] | `____` |
| min_samples_split | [2, 5, 10] | `____` |

**Best CV F1:** `____`

| Metric | Fake (0) | Real (1) | Weighted Avg |
|--------|----------|----------|--------------|
| Precision | `____` | `____` | `____` |
| Recall | `____` | `____` | `____` |
| F1-Score | `____` | `____` | `____` |
| Accuracy | | | `____` |

**Confusion Matrix:**
```
Predicted ->   Fake    Real
Actual Fake  [ ____   ____ ]
Actual Real  [ ____   ____ ]
```

---

## 5. Final Comparison Table

| # | Model | Accuracy | F1 | Precision | Recall |
|---|-------|----------|----|-----------|--------|
| 1 | K-Nearest Neighbor | `____` | `____` | `____` | `____` |
| 2 | Logistic Regression | `____` | `____` | `____` | `____` |
| 3 | SVM (Linear) | `____` | `____` | `____` | `____` |
| 4 | K-Means Clustering | `____` | `____` | `____` | `____` |
| 5 | Neural Network | `____` | `____` | `____` | `____` |
| 6 | Gradient Boosting | `____` | `____` | `____` | `____` |

**Best Performing Model:** `____________________`
**Best F1 Score:** `____`

---

## 6. Analysis and Observations

- **KNN:** Sensitive to high dimensionality. Cosine distance works better for text.
- **Logistic Regression:** Typically strong on sparse TF-IDF features. L2 regularization prevents overfitting.
- **SVM (Linear):** Well-suited for high-dimensional sparse text. Often best-performing for text classification.
- **K-Means:** Unsupervised; expected weaker performance than supervised models.
- **Neural Network:** Can capture non-linear patterns with proper tuning.
- **Gradient Boosting:** Ensemble of weak learners; strong but slower to train.

### Preprocessing Impact
- Keeping negation words (not, no, never) was critical to preserve post meaning.
- Emoji-to-text conversion preserved sentiment (e.g., mask emoji -> wearing_face_mask).
- Hashtag text extraction retained stance signals (e.g., #FakeNews -> fakenews).

---

## 7. Submission Checklist

- [ ] RollNo_Task1_PrepareDataset.py
- [ ] RollNo_Task2_Preprocessing.py
- [ ] RollNo_Task3_TFIDF.py
- [ ] RollNo_KNN.py
- [ ] RollNo_LogisticRegression.py
- [ ] RollNo_SVM.py
- [ ] RollNo_KMeans.py
- [ ] RollNo_NeuralNetwork.py
- [ ] RollNo_GradientBoosting.py
- [ ] RollNo_Task5_Evaluation.py
- [ ] data/splits/train.csv
- [ ] data/splits/val.csv
- [ ] data/splits/test.csv
- [ ] RollNo_Report.pdf

---
*Report for ML Home Assignment - Constraint@AAAI-2021 COVID-19 Fake News Detection*