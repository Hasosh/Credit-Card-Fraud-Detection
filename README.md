# Credit Card Fraud Detection

## Table of Contents
1. Project Description
2. Dataset
3. Code
4. Further Reading
5. References

## Project Description

This project aims to tackle credit card fraud by implementing anomaly detection techniques on tabular data. 
The objective is to identify fraudulent transactions, which are typically rare but hold significant financial implications. 
Using deep learning to automate fraud detection can vastly reduce the manual labor involved, and potentially save millions of dollars in fraud losses. 
The dataset for this project comprises various transaction attributes.

├── LICENSE

├── README.md

├── requirements.txt

├── config          # Configuration files for training

├── data            

│   └── runs        # Artifacts created during learning

├── doc             # Some documentation, presentations, etc.  

├── scripts         # Helper scripts

├── src             # Source code

│   ├── main.py     # Main Script

│   ├── curriculum  # Curriculum Learning Implementation

│   ├── data        # Data Handling and Generation

│   ├── experiments # Experiment Implementations

│   ├── loss        # Loss Implementations

│   ├── models      # Model Implementations

│   └── utils       # Utility Functions

└── tmp             # Temporary Files

## Dataset

The [Kaggle: Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data) features over 550,000 European credit card transactions, aimed at binary classification to identify fraudulent activities. 
It includes a unique identifier and 28 anonymized features, which represent various transaction attributes like time and location, along with the transaction amount for each transaction.
For computational reasons, our study uses a modified version of this dataset with fewer samples.
Completely unsupervised, your training data will contain **only** normal data.

## Code

This project includes steps for data loading, exploratory data analysis (EDA), and shallow and deep learning models for fraud detection.

### Quickstart

This is a quick and easy guide to run our code.

#### Requirements

First, you have to install all the needed packages.
You can do this by typing the following command in your terminal:

## Further Reading
Here you can find our [written report]() and our [final presentation]().

## Credits
We are Hasan Evci and Tareq Abu El Komboz, M.Sc. Computer Science students at the University of Stuttgart.
You can find us on [Hasans-Github](https://github.com/Hasosh), [Tareqs-Gihub](https://github.com/TareqKomboz) and [Hasans-LinkedIn](https://www.linkedin.com/in/hasan-evci-41089922b/) and [Tareqs-LinkedIn](www.linkedin.com/in/tareqkomboz).

## References
- [Lukas Ruff, Jacob R. Kauffmann, Robert A. Vandermeulen, Grégoire Montavon, Wojciech Samek, Marius Kloft,
Thomas G. Dietterich, and Klaus-Robert Müller. A unifying review of deep and shallow anomaly detection. Proc.
IEEE, 109(5):756–795, 2021.](https://arxiv.org/abs/2009.11732)
- [Charu C. Aggarwal. An Introduction to Outlier Analysis, pages 1–34. Springer International Publishing, Cham,
2017.](https://link.springer.com/chapter/10.1007/978-3-319-47578-3_1)
- [Raghavendra Chalapathy, Aditya Krishna Menon, and Sanjay Chawla. Anomaly detection using one-class neural
networks. CoRR, abs/1802.06360, 2018.](https://arxiv.org/abs/1802.06360)
