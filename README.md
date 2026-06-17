# Credit Card Fraud Detection

> Supervised ML to flag fraudulent transactions in a highly imbalanced dataset (0.172% fraud), comparing four classifiers on precision, recall, F1, and ROC-AUC — the metrics that actually matter when fraud is rare.

**Stack:** Python · scikit-learn · pandas · NumPy · Matplotlib
**Domain:** Fintech · Fraud / Risk · Imbalanced classification

---

## Problem
Credit-card fraud causes billions in annual losses, and rule-based systems struggle to keep up. The core challenge is **extreme class imbalance** — fraud is 0.172% of transactions — so raw accuracy is misleading and recall/precision/AUC must drive evaluation.

## Data
[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — **284,807 transactions, 492 fraudulent (0.172%)**.

## Approach
- Preprocessing: dropped `Time`, normalized `Amount`, Min-Max scaling (KNN is scale-sensitive), z-score handling.
- Trained and tuned four classifiers: **K-Nearest Neighbors, Logistic Regression, Decision Tree, SVM (RBF kernel)**.
- Evaluated with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 99.94% | 88.67% | 82.94% | 85.71% | 93.5% |
| Logistic Regression | 91.37% | 93.54% | 88.77% | 91.09% | 95.1% |
| Decision Tree | 99.91% | 74.28% | 75.91% | 75.09% | 96.8% |
| **SVM (RBF)** | 93.58% | **95.87%** | 86.11% | **90.73%** | **97.3%** |

**Takeaway:** SVM (RBF) gave the best precision/F1 balance for fraud detection; Logistic Regression was the most balanced lightweight model. High accuracy alone (KNN/DT ~99.9%) is deceptive under imbalance — the precision/recall/AUC view tells the real story.

## My role
Built and tuned the **KNN and Logistic Regression** models, performed EDA and dataset preprocessing, and authored project documentation. (Team project, ECE 537 Data Mining — teammate built Decision Tree & SVM.)

## Report
Full methodology and results: `Final_Project_report_Group_4.pdf`.
