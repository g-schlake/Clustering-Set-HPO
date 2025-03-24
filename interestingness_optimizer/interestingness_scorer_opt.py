import numpy as np
import optunity
from tqdm import tqdm

from auxiliaries.misc import generationMethod
from interestingness_optimizer.interestingness_scorer import InterestingnessScorer


class InterestingnessScorerOptunity(InterestingnessScorer):

    def __init__(self, validation_functions, algorithms, data, similarity_function, interestingness_function,
                 method=generationMethod.BINARY, time_budget=20):  # , num_algs=10):
        super().__init__(validation_functions, algorithms, data, similarity_function, interestingness_function, method)
        # self.num_algs = num_algs
        self.name = "Optunity"
        self.time_budget = time_budget
        search = {}
        if method == generationMethod.FIXED:
            self.slots = []
            for alg, number in algorithms:
                for i in range(number):
                    for key, value in self.algorithm_dict[alg.name][0].hyperparams.items():
                        search[f"{alg.name}_{i}_{key}"] = value
                    self.slots.append(f"{alg.name}_{i}")
        elif method == generationMethod.BINARY:
            for alg, number in algorithms:
                for i in range(number):
                    search[f"{alg.name}_{i}"] = {"False": None, "True": {}}
                    for key, value in self.algorithm_dict[alg.name][0].hyperparams.items():
                        search[f"{alg.name}_{i}"]["True"][f"{alg.name}_{i}_{key}"] = value
        elif method == generationMethod.ALG_NUMBER:
            for alg, number in algorithms:
                search[alg.name] = {}
                for i in range(number + 1):
                    search[alg.name][str(i)] = {
                        f"{alg.name}_{i}_{key}_{j}": value for key, value in
                        self.algorithm_dict[alg.name][0].hyperparams.items() for j in range(i)
                    }
                    if i == 0:
                        search[alg.name]["0"] = None
        else:
            raise ValueError("Method must be 0,1 or 2")
        self.search = search

    def process_hparams(self, **kwargs):
        if self.method == generationMethod.FIXED:
            return self.process_fixed(**kwargs)
        elif self.method == generationMethod.BINARY:
            return self.process_binary(**kwargs, true_signal="True")
        elif self.method == generationMethod.ALG_NUMBER:
            return self.process_alg_number(**kwargs)
        else:
            raise ValueError
        hyperparams = [{key: value for key, value in dict(kwargs).items() if int(key[-1]) == i} for i in
                       range(self.num_algs)]
        for i in range(len(hyperparams)):
            hparams = hyperparams[i]
            alg = self.algorithm_dict[hparams[f"algorithm_{i}"]]
            for hparam in hparams:
                if hparam.startswith("algorithm"):
                    new_name = param_name = hparam
                elif hparam.startswith(alg.name):
                    param_name = hparam[len(alg.name) + 1:-2]
                    new_name = f"{alg.name}_{param_name}_{i}"
                else:
                    continue
                if param_name in alg.integer_params:
                    out_value = int(hparams[hparam])
                else:
                    out_value = hparams[hparam]
                out_dict[new_name] = out_value
        return out_dict

    def run(self, desc=None):
        if not desc:
            desc = f"Optunity {[x().name for x in self.validation_functions]} {[x[0].name for x in self.algorithms]}"
        self.t = tqdm(total=self.time_budget, desc=desc,
                      unit="configuration", leave=False)
        res = optunity.maximize_structured(self.score, num_evals=self.time_budget, search_space=self.search)

        # res = optunity.maximize(self.score, num_evals = self.time_budget, solver_name=self.solver_name, **self.hyperparams)
        params = res[0]
        params_new = {}
        for param in params:
            split = param.split("|")
            param_new = split[-1]
            params_new[param_new] = params[param]
        params = self.process_hparams(**params_new)
        if len(params) == 0:
            return [], -np.inf, None, None, {}
        best_result = self.generate_clusterings(params)
        sim_mat, val_mat = self.evaluate_clusterings(best_result)
        score = self.interestingness_function(sim_mat, val_mat)
        return [x[1] for x in best_result], score, sim_mat, val_mat, params
