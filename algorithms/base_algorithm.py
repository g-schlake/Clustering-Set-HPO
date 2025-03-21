import abc

import numpy as np
import sklearn


class algorithm(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.data = None
        self.labels = None
        self.parameter_bounds = None
        self.integer_params = None
        self.preprocessing_params = None
        self.name = None
        self.distances = None
        self.distance = None
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    def ensure_bounds(self, **kwargs):
        for param, limits in self.parameter_bounds.items():
            if len(limits) == 2 or (limits[2] and limits[3]):
                if kwargs[param] < limits[0]:
                    kwargs[param] = limits[0]
                elif kwargs[param] > limits[1]:
                    kwargs[param] = limits[1]
            else:
                eps = 1
                if param not in self.integer_params:
                    eps = np.finfo(float).eps
                if not limits[2]:
                    if kwargs[param] <= limits[0]:
                        kwargs[param] = limits[0] + eps
                if not limits[3]:
                    if kwargs[param] >= limits[1]:
                        kwargs[param] = limits[1] - eps
        for param in self.integer_params:
            kwargs[param] = int(kwargs[param])
        return kwargs

    def constrained_distance_matrix(self, data, linked_samples, unlinked_samples, distance, inf_value='Max'):
        distance_matrix = sklearn.metrics.pairwise_distances(data, metric=distance, n_jobs=-1)
        for cluster in linked_samples:
            for i in cluster:
                for j in cluster:
                    distance_matrix[i, j] = 0
        max_distance = np.max(distance_matrix)
        for cluster in unlinked_samples:
            for i in cluster:
                for j in cluster:
                    distance_matrix[i, j] = 2 * max_distance
        return distance_matrix
