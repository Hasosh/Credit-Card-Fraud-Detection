# models.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest

class NaiveBaseline:
    def predict(self, X):
        return [True] * len(X)


class KMeansAnomalyDetector: # only works for n_clusters=2
    def __init__(self, n_clusters=2, n_init=10, random_state=0):
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
        self.centroid_train = None
        self.normal_cluster = None

    def fit(self, X_train):
        self.centroid_train = np.mean(X_train, axis=0)
        self.model.fit(X_train)
        centroids_test = self.model.cluster_centers_
        distances = np.linalg.norm(centroids_test - self.centroid_train, axis=1)
        self.normal_cluster = np.argmin(distances)

    def predict(self, X_test):
        labels = self.model.predict(X_test)
        anomaly_cluster = 1 - self.normal_cluster
        return (labels == anomaly_cluster).astype(int)


class DBSCANAnomalyDetector:
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X):
        return self.model.fit_predict(X)


class OCSVM:
    def __init__(self, nu=0.5, kernel='rbf', gamma='auto'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class SGDOCSVM:
    def __init__(self):
        self.model = SGDOneClassSVM()

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class KNNAnomalyDetector:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.threshold = None

    def fit(self, X):
        self.model.fit(X)

    def compute_threshold(self, X, percentile=95):
        distances, _ = self.model.kneighbors(X)
        self.threshold = np.percentile(distances[:, self.n_neighbors - 1], percentile)

    def predict(self, X):
        distances, _ = self.model.kneighbors(X)
        return (distances[:, self.n_neighbors - 1] > self.threshold).astype(int)


class LOFAnomalyDetector:
    def __init__(self, n_neighbors=20, contamination='auto'):
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def fit_predict(self, X):
        return self.model.fit_predict(X)


class IsolationForestDetector:
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=42):
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, 
                                     contamination=contamination, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

