from algorithms import dbscan, kmeans, meanshift

registered_algorithms = [dbscan.Dbscan, kmeans.Kmeans, meanshift.Meanshift]


def get_names_arbitrary_shape():
    return ["HDBSCAN", "DBSCAN", "OPTICS"]


def get_algorithms():
    ret = []
    for x in registered_algorithms:
        alg = x()
        ret.append(alg)
    return ret
