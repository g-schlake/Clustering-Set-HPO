import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

from algorithms import base_algorithm


class Meanshift(base_algorithm.algorithm):
    def __init__(self, **kwargs):
        self.alg = MeanShift()
        self.name = "MeanShift"
        self.preprocessing_params = None
        self.hyperparams = {"bandwidth": [0.1, 50]}
        self.parameter_bounds = {"bandwidth": [0, np.inf, False, True]}
        self.integer_params = []
        self.kwargs = kwargs

    def fit(self, data, bandwidth):
        args = self.ensure_bounds(bandwidth=bandwidth)
        estimator = MeanShift(**args, n_jobs=32, cluster_all=False, **self.kwargs)
        estimator.fit(data)
        return estimator
