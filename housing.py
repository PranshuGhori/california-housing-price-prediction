"""Data loading and the preprocessing pipeline for the California housing notebook."""

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("datasets", "housing")
TARGET = "median_house_value"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download and extract housing.tgz once; later calls reuse the local copy."""
    csv_path = os.path.join(housing_path, "housing.csv")
    if os.path.exists(csv_path):
        return csv_path
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as tgz:
        tgz.extractall(path=housing_path, filter="data")
    return csv_path


def load_housing_data(housing_path=HOUSING_PATH):
    return pd.read_csv(fetch_housing_data(housing_path=housing_path))


def income_category(median_income):
    """Bucket median income for stratified splitting."""
    return pd.cut(median_income, bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])


# --- feature engineering -----------------------------------------------------

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """RBF similarity of each district to KMeans centres fitted on (lat, lon)."""

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


def make_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        transformers=[
            ("bedrooms_ratio", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline(),
             ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
            ("geo", ClusterSimilarity(n_clusters, gamma, random_state), ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
        sparse_threshold=0,
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bootstrap_rmse_ci(y_true, y_pred, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for RMSE, resampling test districts with replacement."""
    rng = np.random.default_rng(seed)
    sq = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    idx = rng.integers(0, len(sq), size=(n_boot, len(sq)))
    boots = np.sqrt(sq[idx].mean(axis=1))
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
