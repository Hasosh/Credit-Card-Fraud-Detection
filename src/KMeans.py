import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl

if __name__ == '__main__':
    # Loading first setup
    with open('../data/setup_1.pkl', 'rb') as f:
        setup1 = pkl.load(f)

    X_train, _, X_test, y_test = setup1['X_train'], setup1['y_train'], setup1['X_test'], setup1['y_test']

    # Assuming the first column is 'id' and the last column is 'amount'
    columns = ['Feature_' + str(i) for i in range(1, X_train.shape[1] - 1)] + ['Amount']
    X_train_df = pd.DataFrame(X_train[:, 1:], columns=columns)  # Excluding 'id'
