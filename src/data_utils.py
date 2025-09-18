"""
Dataset loading and preprocessing for quantum feature map experiments.
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def scale_features(X, range_min=0, range_max=np.pi):
    """Scale features to [0, pi] for use as rotation angles."""
    scaler = MinMaxScaler(feature_range=(range_min, range_max))
    return scaler.fit_transform(X)


def load_moons(n_samples=200, noise=0.15, seed=42):
    """make_moons dataset scaled to [0, pi]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = scale_features(X)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def load_circles(n_samples=200, noise=0.1, seed=42):
    """make_circles dataset scaled to [0, pi]."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    X = scale_features(X)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def load_iris_2d(seed=42):
    """Iris 2 features, 2 classes, scaled to [0, pi]."""
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target
    mask = y < 2
    X, y = X[mask], y[mask]
    X = scale_features(X)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def load_breast_cancer_pca(n_components=4, seed=42):
    """Breast cancer dataset reduced via PCA, scaled to [0, pi]."""
    data = load_breast_cancer()
    pca = PCA(n_components=n_components, random_state=seed)
    X = scale_features(pca.fit_transform(data.data))
    return train_test_split(X, data.target, test_size=0.25, random_state=seed, stratify=data.target)
