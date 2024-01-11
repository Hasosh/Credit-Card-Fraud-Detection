import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler

class MyDataLoader:
    def __init__(self):
        pass

    def load_head(self, filepath):
        df_head = pd.read_csv(filepath)
        return df_head

    def load_setup(self, setup_path):
        with open(setup_path, 'rb') as f:
            return pkl.load(f)
        
    def transform_data(self, X, is_train=False):
        if is_train:
            self.scaler = StandardScaler().fit(X[:, -1].reshape(-1, 1))
        X[:, -1] = self.scaler.transform(X[:, -1].reshape(-1, 1)).flatten()
        return X[:, 1:]  # Exclude 'id' column