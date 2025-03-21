import numpy as np

from measures import base_measure
from measures.auxiliaries.rebuild_dbcv import rebuild_dbcv


class DBCV(base_measure.BaseMeasure):
    def __init__(self):
        super().__init__()
        self.name = "DBCV"
        self.worst_value = -1
        self.best_value = 1
        self.kwargs = {}
        self.needs_quadratic = False

    def score(self, data, labels):
        return self.ensure_finite(rebuild_dbcv(data, labels))

    def score_norm(self, data, labels):
        return self.score(data, labels)

