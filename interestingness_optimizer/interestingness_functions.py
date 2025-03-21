import numpy as np
from auxiliaries.misc import finite_number


def naive_interestingness(similarity_matrix, validation_matrix):
    similarity_matrix = similarity_matrix - np.eye(similarity_matrix.shape[0])
    val_sums = np.sum(validation_matrix, axis=1)
    sim_maxs = np.max(similarity_matrix, axis=1)
    res = finite_number(np.mean(val_sums) - np.mean(sim_maxs))
    return res


def naive_interestingness_avg(similarity_matrix, validation_matrix):
    similarity_matrix = similarity_matrix - np.eye(similarity_matrix.shape[0])
    val_sums = np.mean(validation_matrix, axis=1)
    sim_maxs = np.max(similarity_matrix, axis=1)
    res = finite_number(np.mean(val_sums) - np.mean(sim_maxs))
    return res


def naive_interestingness_mean_sim(similarity_matrix, validation_matrix):
    similarity_matrix = similarity_matrix - np.eye(similarity_matrix.shape[0])
    val_sums = np.mean(validation_matrix, axis=1)
    sim_maxs = np.mean(similarity_matrix, axis=1)
    res = finite_number(np.mean(val_sums) - np.mean(sim_maxs))
    return res


def naive_interestingness_min_sim(similarity_matrix, validation_matrix):
    similarity_matrix = similarity_matrix - np.eye(similarity_matrix.shape[0])
    val_sums = np.mean(validation_matrix, axis=1)
    sim_maxs = np.min(similarity_matrix, axis=1)
    res = finite_number(np.mean(val_sums) - np.mean(sim_maxs))
    return res


def mmr_interestingness(similarity_matrix, validation_matrix, beta=0.5):
    similarity_matrix = similarity_matrix - np.eye(similarity_matrix.shape[0])
    num_clus = similarity_matrix.shape[0]
    rel = (num_clus - 1) * np.sum(validation_matrix)
    div = np.sum(np.sum(similarity_matrix, axis=1), axis=0)
    return (1 - beta) * rel + beta * div
