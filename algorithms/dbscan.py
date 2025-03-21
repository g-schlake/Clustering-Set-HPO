import numpy as np
import sklearn.metrics.pairwise
from sklearn.cluster import DBSCAN

from algorithms import base_algorithm
from measures.dbcv_measures import DBCV


class Dbscan(base_algorithm.algorithm):
    def __init__(self):
        self.alg = DBSCAN()
        self.preprocessing_params = None
        self.name = "DBSCAN"
        self.hyperparams = {"eps": [0.1, 4], "min_samples": [3, 20]}
        self.parameter_bounds = {"min_samples": [2, np.inf, True, True], "eps": [0, np.inf, False, True]}
        self.integer_params = ["min_samples"]

    def fit(self, data, eps, min_samples):
        # data = sklearn.metrics.pairwise_distances(data, n_jobs=-1)
        data = sklearn.metrics.pairwise_distances(data, n_jobs=-1)
        args = self.ensure_bounds(eps=eps, min_samples=min_samples)
        estimator = DBSCAN(**args, n_jobs=32, metric="precomputed")
        estimator.fit(data)
        return estimator
