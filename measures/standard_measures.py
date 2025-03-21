import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score

from evaluations.dataset_fetcher import clean_labels
from measures import base_measure


class Silhouette_Coefficient(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "SWC"
        self.worst_value = -1
        self.best_value = 1
        self.function = silhouette_score
        self.function_norm = silhouette_score
        self.kwargs = {"metric": "precomputed"}
        self.needs_quadratic = True


class VRC(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "VRC"
        self.worst_value = -1 * np.inf
        self.best_value = 0
        self.function = calinski_harabasz_score
        self.function_norm = ValueError
        self.kwargs = {}
        self.needs_quadratic = False


class AMI(base_measure.BaseMeasure):
    def __init__(self, labels):
        super().__init__()
        self.name = "AMI"
        self.worst_value = -1
        self.function_norm = ValueError
        self.function = NotImplementedError
        self.function_clusters = ValueError
        self.kwargs = {}
        self.needs_quadratic = False
        self.less_is_better = True
        self.labels = labels

    def score(self, data, labels: np.ndarray, **kwargs) -> float:
        ground_truth = self.labels
        outlier_indxs = np.where(labels == -1)[0]
        new_labels = np.array(range(len(outlier_indxs))) + np.max(labels) + 1
        labels[outlier_indxs] = new_labels
        alt_gt = clean_labels(ground_truth)
        outlier_indxs_gt = np.where(alt_gt == -1)[0]
        new_gt = np.array(range(len(outlier_indxs_gt))) + np.max(alt_gt) + 1
        alt_gt[outlier_indxs_gt] = new_gt
        return adjusted_mutual_info_score(labels, alt_gt)

    def score_norm(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        return self.score(data, labels)

    def score_distance_function(self, data: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        return self.score(data, labels)


def ofami(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    ground_truth = labels_b
    outlier_indxs = np.where(labels_a == -1)[0]
    new_labels = np.array(range(len(outlier_indxs))) + np.max(labels_a) + 1
    labels_a[outlier_indxs] = new_labels
    alt_gt = clean_labels(ground_truth)
    outlier_indxs_gt = np.where(alt_gt == -1)[0]
    new_gt = np.array(range(len(outlier_indxs_gt))) + np.max(alt_gt) + 1
    alt_gt[outlier_indxs_gt] = new_gt
    return adjusted_mutual_info_score(labels_a, alt_gt)
