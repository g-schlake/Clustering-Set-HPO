import numpy as np
from sklearn.cluster import KMeans
from algorithms import base_algorithm


class Kmeans(base_algorithm.algorithm):
    def __init__(self, **kwargs):
        self.alg = KMeans()
        self.preprocessing_params = None
        self.name = "KMeans"
        self.kwargs = kwargs
        self.hyperparams = {"n_clusters": [2, 30]}
        self.parameter_bounds = {"n_clusters": [2, np.inf, True, True]}
        self.integer_params = ["n_clusters"]


    def fit(self, data, n_clusters):
        args = self.ensure_bounds(n_clusters=n_clusters)
        estimator = KMeans(**args, n_init="auto", **self.kwargs)
        estimator.fit(data)
        return estimator
