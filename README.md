# Credit Card Fraud Detection

## Table of Contents
1. Project Description
2. Dataset
3. Code
4. Further Reading
5. References

## Overview

This is part of the module "Laboratory Course Artificial Intelligence: Deep Learning Lab 2023/2024".
Thanks to our supervisor [Rodrigo](https://github.com/RodrigoLPA) for his support!

## Project Description

In this project, we aim to tackle credit card fraud by implementing anomaly detection techniques on tabular data. 
The objective is to identify fraudulent transactions, which are typically rare but hold significant financial implications. 
Using deep learning to automate fraud detection can vastly reduce the manual labor involved, and potentially save millions of dollars in fraud losses. 

This study reveals that deep learning models generally outperform traditional methods. 
This suggests their greater efficacy, especially in dealing with the challenges posed by high-dimensional data. 
The One Class Neural Network emerges as the most effective model, with significant proficiency in identifying fraudulent transactions
Interestingly, traditional models like kNN distance and SGD OC-SVM show specific strengths in certain metrics, suggesting their continued relevance. 
The similar performance of various autoencoder models suggests that data noise levels are low and that sophisticated models like DAEs and VAEs do not necessarily offer additional benefits in this context.

## Dataset

The [Kaggle: Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data) features over 550,000 European credit card transactions, aimed at binary classification to identify fraudulent activities. 
It includes a unique identifier and 28 anonymized features, which represent various transaction attributes like time and location, along with the transaction amount for each transaction.
For computational reasons, our study uses a modified version of this dataset with fewer samples.
Completely unsupervised, your training data will contain **only** normal data.

## Main Results

Our main results are compactly presented in the following Table. 
Every row corresponds to a different model and every column corresponds to one metric. 
For each metric, we highlighted the model that performed best in bold font.
The best-performing model is the DL model OC-NN, whereas the worst-performing is the Isolation Forest.
The best baseline model is the Mahalanobis distance model.

<figure>
  <img src="https://github.com/Hasosh/Credit-Card-Fraud-Detection/blob/master/img/main_results.png" alt="Table of main results"/>
</figure>

## Code

This project includes steps for data loading, exploratory data analysis (EDA), and shallow and deep learning model training for fraud detection.

### Project Structure

```bash
.
├── README.md
├── requirements.txt
├── data            		# Dataset used for training and evaluation
├── doc             		# Additional documents (Written report, Final presentation) 
│	├── paper.pdf     	
│	├── presentation.pdf	
├── img             		# Images of experiments and results
├── model_saves          	# Weights for our best Deep Learning models
├── notebooks       		# Helper notebooks
├── src/main           		# Source code
	├── config.py  						
	├── data_loader.py        				
	├── dl_models.py 					
	├── evaluation.py        				
	├── experiments_autoencoder.py      			 
	├── experiments_ocnn.py     				 
	├── functionality_and_baseline_model_training.ipynb  	 
	├── kfold_ocnn.py        				 
	├── models.py 						 
	├── train_autoencoder.py        			 
	├── train_ocnn.py      					 
	└── utils.py
```

### Quickstart

This is a quick and easy guide to run our code.

#### Requirements

To run the project, you need to have Python 3.9 or higher installed. To install the required packages, you can type in the following command in your terminal:

```
pip install -r requirements.txt
```

For GPU acceleration within the PyTorch framework, it's necessary to install CUDA tailored for PyTorch. To enable this feature, please follow the instructions provided [here](https://pytorch.org/get-started/locally/).

#### Running experiments

Before running experiments, you should first specify the hyperparameters in the `config.py` file which can be found in the `src/main` folder. Examples: 
- to specify which data to use, configure the `DATA_PATH` parameter.
- if you want to run autoencoder experiments, you have to specify the type of autoencoder, e.g. `'Autoencoder'`, `'DenoisingAE'`. For custom architectures (that you can modify in the `dl_models.py` script), you can use `'CustomAE'`, `'CustomDAE'`, `'CustomVAE'`.
- if you do not want to create runs on wandb, you should set `USE_WAND=False`.

There are different scripts to run experiments (for different purposes):
- `train_autoencoder.py` (training one autoencoder architecture, i.e. AE, DAE, VAE)
- `train_ocnn.py` (training one OC-NN)
- `kfold_autoencoder.py` (k-fold cross-validation for autoencoder and reporting macro-averaged results)
- `kfold_ocnn.py` (k-fold cross-validation for OC-NN and reporting macro-averaged results)
- `experiments_autoencoder.py` (our conducted experiments for the autoencoder)
- `experiments_ocnn.py` (our conducted experiments for the OC-NN)

To run one of these scripts, you can simply type in the command:

```
python src/main/[SCRIPT_NAME]
```

## Further Reading
Here you can find our [written report](https://github.com/Hasosh/Credit-Card-Fraud-Detection/blob/master/doc/paper.pdf) and our [final presentation](https://github.com/Hasosh/Credit-Card-Fraud-Detection/blob/master/doc/presentation.pdf).

## Credits
We are Hasan Evci and Tareq Abu El Komboz, M.Sc. Computer Science students at the University of Stuttgart.
You can find us on [Hasans-Github](https://github.com/Hasosh), [Tareqs-Gihub](https://github.com/TareqKomboz) and [Hasans-LinkedIn](https://www.linkedin.com/in/hasan-evci-41089922b/) and [Tareqs-LinkedIn](https://www.linkedin.com/in/tareqkomboz/).

## References
- [Lukas Ruff, Jacob R. Kauffmann, Robert A. Vandermeulen, Grégoire Montavon, Wojciech Samek, Marius Kloft,
Thomas G. Dietterich, and Klaus-Robert Müller. A unifying review of deep and shallow anomaly detection. Proc.
IEEE, 109(5):756–795, 2021.](https://arxiv.org/abs/2009.11732)
- [Charu C. Aggarwal. An Introduction to Outlier Analysis, pages 1–34. Springer International Publishing, Cham,
2017.](https://link.springer.com/chapter/10.1007/978-3-319-47578-3_1)
- [Raghavendra Chalapathy, Aditya Krishna Menon, and Sanjay Chawla. Anomaly detection using one-class neural
networks. CoRR, abs/1802.06360, 2018.](https://arxiv.org/abs/1802.06360)
