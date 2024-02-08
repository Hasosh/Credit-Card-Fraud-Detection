# Credit Card Fraud Detection

## Abstract
Improving algorithms for credit card fraud detection can save institutions and private persons tremendous amounts of money and is of huge importance. 
In this work, we show that deep learning methods are superior to many traditional anomaly detection methods in the field of credit card fraud detection.
Using the Credit Card Fraud Detection Dataset 2023 with over 550,000 credit card transactions, we found that nearly all proposed deep learning architectures achieved better metrics than traditional baselines, suggesting their greater efficacy for high-dimensional data. 
The best-performing model overall is the one-class neural network with a macro-average F1-score of 93%, whereas the best-performing baseline model is the kNN distance method with a macro-average F1-score of 89%. 
Interestingly, the kNN distance method slightly outperforms advanced variations of the autoencoder, indicating that simpler models can still be highly effective in certain scenarios, particularly in datasets where the distinction between normal and anomalous data is more pronounced. 
This finding emphasizes the importance of considering both advanced and traditional approaches in fraud detection strategies.

## Overview

### Description

This project aims to tackle credit card fraud by implementing anomaly detection techniques on tabular data. 
The objective is to identify fraudulent transactions, which are typically rare but hold significant financial implications. 
Using deep learning to automate fraud detection can vastly reduce the manual labor involved, and potentially save millions of dollars in fraud losses. 
The dataset for this project comprises various transaction attributes.

### Motivation

Credit card fraud is a growing concern with the rise in digital transactions.
The financial sector spends a significant amount of resources on fraud prevention and detection. 
Automating this process using deep learning can significantly cut costs, improve accuracy, and speed up the fraud detection process. 
The impact of a successful implementation could be substantial, making transactions safer and saving financial institutions immense resources.

### Evaluation

Evaluating anomaly detection models has unique challenges due to the imbalanced nature of the data. 
Fraudulent transactions are significantly outnumbered by genuine transactions, which makes it difficult for models to learn the characteristics of fraudulent behavior effectively. 
Moreover, traditional accuracy as a metric can be misleading since a model that predicts every transaction as genuine could still achieve high accuracy due to the data imbalance.

In this light, the evaluation of the project should focus on metrics better suited for imbalanced datasets and anomaly detection tasks. 
Here are some examples of such metrics:

- **Precision**: The ratio of correctly identified fraudulent transactions to the total identified as fraudulent. High precision indicates a lower false positive rate which is crucial to avoid alarming users unnecessarily.
- **Recall**: The ratio of correctly identified fraudulent transactions to the total actual fraudulent transactions. High recall is important to catch as many fraudulent transactions as possible.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics. This is especially useful in the context of an imbalanced dataset.
- **Area Under the Precision-Recall Curve (AUC-PR)**: Unlike AUC-ROC, AUC-PR is more informative when dealing with imbalanced datasets, as it focuses on the performance of the positive (minority) class.
- **Matthews Correlation Coefficient (MCC)**: MCC is a balanced metric that takes into account true and false positives and negatives. It is a reliable indicator of the quality of binary classifications, especially useful in imbalanced datasets.

The trade-off between precision and recall is pivotal in fraud detection. 
Aiming for high precision could result in missing some fraudulent transactions (lower recall), while aiming for high recall could result in many false positives (lower precision). 
The choice between focusing on precision or recall would depend on the specific costs associated with false positives and false negatives in the given financial context. 
For this project, you will decide what is an acceptable failure rate and justify it.

Baseline models should include simple statistical anomaly detection methods or basic machine learning models like logistic regression and One-Class SVMs.

Your result analysis should include at least the following:

- **Precision-Recall Curve**: A plot of Precision vs Recall to understand the trade-off between the two metrics and to compare different models.
- **Confusion Matrix**: Visualizing the true positives, true negatives, false positives, and false negatives to get a clear picture of the model performance.
- **Table comparing baselines against your models.**

## Data

### Files

You will work on a variation of the following dataset: [Kaggle: Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data). The features will be the same, but the experimental setup will be different.

#### Case 1

Completely unsupervised, your training data will contain **only** normal data.

## Code

### Getting Started Notebook

A Jupyter notebook will be provided to help kickstart the project. This notebook will include steps for data loading, initial exploratory data analysis (EDA), and example models for fraud detection. The link or file path to the notebook will be shared once it's prepared.
