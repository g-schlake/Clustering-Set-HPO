
from measures.dbcv_measures import DBCV
from measures.standard_measures import Silhouette_Coefficient

registered_measures = [Silhouette_Coefficient, DBCV]


def get_measures():
    return registered_measures
