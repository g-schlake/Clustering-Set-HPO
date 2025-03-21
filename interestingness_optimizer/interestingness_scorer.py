import abc
import itertools
import warnings
from multiprocessing import Pool
import numpy as np
from auxiliaries.misc import finite_number
import os


def generate_clustering(args):
    idx, alg, data, hparams = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = alg.fit(data, **hparams).labels_
    return idx, labels


def generate_validation(data, labels, validation_function):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = validation_function().score_norm(data, labels[1])
    return labels[0], validation_function().name, value


def generate_similarity(similarity_function, labels_a, labels_b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = similarity_function(labels_a[1], labels_b[1])
    return (labels_a[0], labels_b[0]), value


class InterestingnessScorer(abc.ABC):

    def __init__(self, validation_functions, algorithms, data, similarity_function, interestingness_function, method):
        self.validation_functions = validation_functions
        self.algorithms = algorithms
        self.data = data
        self.similarity_function = similarity_function
        self.interestingness_function = interestingness_function
        self.method = method
        self.validation_map = dict(
            itertools.zip_longest([x().name for x in validation_functions], range(len(validation_functions))))
        self.algorithm_dict = {x[0].name: (x[0], x[1]) for x in self.algorithms}
        self.algorithm_names = [x[0].name for x in self.algorithms]

    def generate_clustering(self, args):
        idx, alg, data, hparams = args
        labels = alg.fit(data, **hparams).labels_
        return labels

    def generate_clusterings(self, args):
        algos = [self.algorithm_dict[arg[0]][0] for arg in args]
        if len(algos) > 0:
            with Pool(max(1, min(len(algos), int(os.cpu_count() / 2)))) as pool:
                results = pool.map(generate_clustering,
                                   [(i, algos[i], self.data, args[i][1]) for i in range(len(algos))])
        else:
            results = []
        return results

    def evaluate_clusterings(self, results):
        validation_computations = list(itertools.product([self.data], results, self.validation_functions))
        with Pool(max(1, min(len(validation_computations), int(os.cpu_count() / 2)))) as pool:
            validations = pool.starmap(generate_validation,
                                       validation_computations)
        similarity_computations = list(itertools.product([self.similarity_function], results, results))
        with Pool(max(1, min(len(similarity_computations), int(os.cpu_count() / 2)))) as pool:
            similarities = pool.starmap(generate_similarity,
                                        similarity_computations)
        similarity_matrix = np.zeros((len(results), len(results)))
        for key, value in similarities:
            similarity_matrix[key] = finite_number(value)
        validation_matrix = np.zeros((len(results), len(self.validation_functions)))
        for idx, mes, val in validations:
            validation_matrix[idx, self.validation_map[mes]] = finite_number(val)
        return similarity_matrix, validation_matrix

    def score(self, **kwargs):
        res = -1
        try:
            kwargs = self.process_hparams(**kwargs)
            results = self.generate_clusterings(kwargs)
            if len(results) > 0:
                similarity_matrix, validation_matrix = self.evaluate_clusterings(results)
                res = self.interestingness_function(similarity_matrix, validation_matrix)
        finally:
            self.update_pbar(res)
            if not np.isfinite(res):
                res = self.validation_functions[0].ensure_finite(res)
            assert np.isfinite(res)
            return res

    def update_pbar(self, res):
        if self.t:
            if self.t.postfix:
                prev_best = float(self.t.postfix.split("=")[1])
            else:
                prev_best = -1
            best_res = max(prev_best, res)
            self.t.set_postfix({"best": best_res})
            self.t.update(1)

    def params_to_int(self, strip_last=False, **kwargs):
        for key in kwargs:
            alg = (splitted := key.split("_"))[0]
            if len(splitted) <= 2:
                continue
            if len(splitted) > 3:
                if strip_last:
                    param = "_".join(splitted[2:])
                else:
                    param = "_".join(splitted[2:-1])
            else:
                param = splitted[2]
            if param in self.algorithm_dict[alg][0].integer_params:
                if kwargs[key]:
                    kwargs[key] = int(kwargs[key])
        return kwargs

    def process_fixed(self, to_int=True, **kwargs):
        if to_int:
            kwargs = self.params_to_int(**kwargs)
        used_hparams = [
            (x.split("_")[0], {key[len(x) + 1:]: value for key, value in kwargs.items() if key.startswith(x + "_")}) for
            x in self.slots]
        return used_hparams

    def process_binary(self, to_int=True, true_signal=True, **kwargs):
        if to_int:
            kwargs = self.params_to_int(**kwargs)
        possible_algos = [key for key in kwargs if len(key.split("_")) == 2]
        used_algos = [key for key in possible_algos if kwargs[key] == true_signal]
        used_hparams = [
            (x.split("_")[0], {key[len(x) + 1:]: value for key, value in kwargs.items() if key.startswith(x + "_")}) for
            x in used_algos]
        return used_hparams

    def process_alg_number(self, to_int=True, short_string=False, **kwargs):
        if to_int:
            kwargs = self.params_to_int(**kwargs, strip_last=True)
        used_algos = {x: int(kwargs[x]) for x in self.algorithm_names}
        used_hyperparams = []
        for algo in used_algos:
            if used_algos[algo] > 0:
                hyperparams = self.algorithm_dict[algo][0].hyperparams.keys()
                for i in range(used_algos[algo]):
                    if short_string:
                        used_hyperparams.append(
                            (algo, {hparam: kwargs[f"{algo}_{i}_{hparam}"] for hparam in hyperparams}))
                    else:
                        used_hyperparams.append((algo,
                                                 {hparam: kwargs[f"{algo}_{used_algos[algo]}_{hparam}_{i}"] for
                                                  hparam in hyperparams}))
        return used_hyperparams
