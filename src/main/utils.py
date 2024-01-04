import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RANDOM_SEED = 0

def create_subset(X, y, subset_size=0.1):
    normal_indices = np.where(y == 0)[0]
    anomaly_indices = np.where(y == 1)[0]

    np.random.seed(RANDOM_SEED)
    normal_subset_indices = np.random.choice(normal_indices, int(len(normal_indices) * subset_size), replace=False)
    anomaly_subset_indices = np.random.choice(anomaly_indices, int(len(anomaly_indices) * subset_size), replace=False)

    subset_indices = np.concatenate([normal_subset_indices, anomaly_subset_indices])
    return X[subset_indices], y[subset_indices]

def tsne_scatter(features, labels, dimensions=2, save_as='graph.png'):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d. Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(features)
    
    # initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    # counting dimensions
    if dimensions == 3: 
        ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Anomaly'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Normal'
    )

    plt.legend(loc='best')
    #plt.savefig(save_as) # storing it to be displayed later
    plt.show()