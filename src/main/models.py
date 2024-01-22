# models.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KernelDensity
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture

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
    
    def anomaly_scores(self, X):
        return self.model.score_samples(X)


class SGDOCSVM:
    def __init__(self, random_state=0):
        self.model = SGDOneClassSVM(random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
    
    def anomaly_scores(self, X):
        return self.model.score_samples(X)


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
        y_pred = (distances[:, self.n_neighbors - 1] > self.threshold).astype(int)
        return y_pred, distances[:, self.n_neighbors - 1]


class LOFAnomalyDetector:
    def __init__(self, n_neighbors=20, contamination='auto'):
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit_predict(X)


class IsolationForestDetector:
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=0):
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, 
                                     contamination=contamination, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
    
    def anomaly_scores(self, X):
        return self.model.score_samples(X)
    
class MahalanobisAnomalyDetector:
    def __init__(self):
        self.mean = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, X):
        # Calculate the mean and inverse covariance of the training data
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        self.cov_inv = np.linalg.inv(cov)

    def compute_threshold(self, X, percentile=95):
        # Calculate Mahalanobis distance for the training set
        distances = [distance.mahalanobis(x, self.mean, self.cov_inv) for x in X]
        self.threshold = np.percentile(distances, percentile)

    def predict(self, X):
        # Calculate Mahalanobis distance for the test set
        distances = [distance.mahalanobis(x, self.mean, self.cov_inv) for x in X]
        # Predict anomalies based on the threshold
        y_pred = (distances > self.threshold).astype(int)
        return y_pred, distances

class GMMAnomalyDetector:
    def __init__(self, n_components=3, random_state=0):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.threshold = None

    def fit(self, X):
        self.model.fit(X)

    def compute_threshold(self, X, std_multiplier=3):
        log_likelihood = self.model.score_samples(X)
        self.threshold = np.mean(log_likelihood) - std_multiplier * np.std(log_likelihood)

    def predict(self, X):
        log_likelihood = self.model.score_samples(X)
        y_pred = (log_likelihood < self.threshold).astype(int)
        return y_pred, log_likelihood

class KDEAnomalyDetector:
    def __init__(self, kernel='gaussian', bandwidth=0.5):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.threshold = None

    def fit(self, X):
        self.model.fit(X)

    def compute_threshold(self, X, std_multiplier=3):
        log_density = self.model.score_samples(X)
        self.threshold = np.mean(log_density) - std_multiplier * np.std(log_density)

    def predict(self, X):
        log_density = self.model.score_samples(X)
        y_pred = (log_density < self.threshold).astype(int)
        return y_pred, log_density