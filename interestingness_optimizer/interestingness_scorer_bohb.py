import bohb.configspace as cs
from bohb import BOHB
from tqdm import tqdm

from auxiliaries.misc import generationMethod
from interestingness_optimizer.interestingness_scorer import InterestingnessScorer


class InterestingnessScorerBOHB(InterestingnessScorer):

    def __init__(self, validation_functions, algorithms, data, similarity_function, interestingness_function,
                 method=generationMethod.BINARY, time_budget=20):
        super().__init__(validation_functions, algorithms, data, similarity_function, interestingness_function, method)
        self.time_budget = time_budget
        cs_list = []
        # hyperparams = self.algorithms.hyperparams
        if method == generationMethod.FIXED:
            self.slots = []
            for alg, alg_num in self.algorithms:
                for i in range(alg_num):
                    hyperparams = alg.hyperparams
                    for key, param in hyperparams.items():
                        if key in alg.integer_params:
                            cs_list.append(cs.IntegerUniformHyperparameter(f"{alg.name}_{i}_{key}", param[0], param[1]))
                        else:
                            cs_list.append(cs.UniformHyperparameter(f"{alg.name}_{i}_{key}", param[0], param[1]))
                    self.slots.append(f"{alg.name}_{i}")
        elif method == generationMethod.BINARY:
            for alg, alg_num in self.algorithms:
                for i in range(alg_num):
                    hyperparams = alg.hyperparams
                    parent_name = f"{alg.name}_{i}"
                    parent = cs.CategoricalHyperparameter(parent_name, [True, False])
                    cs_list.append(parent)
                    for key, param in hyperparams.items():
                        child_name = f"{parent_name}_{key}"
                        if key in alg.integer_params:
                            cs_list.append((present := cs.IntegerUniformHyperparameter(child_name, param[0], param[1],
                                                                                       parent == True)))
                        else:
                            cs_list.append(
                                (present := cs.UniformHyperparameter(child_name, param[0], param[1], parent == True)))
                        cs_list.append(cs.CategoricalHyperparameter(f"{child_name}", [None], ~present.cond))
        elif method == generationMethod.ALG_NUMBER:
            for alg, alg_num in self.algorithms:
                cs_list.append((parent := cs.IntegerUniformHyperparameter(f"{alg.name}", 0, alg_num)))
                for i in range(alg_num):
                    hyperparams = alg.hyperparams
                    for key, param in hyperparams.items():
                        if key in alg.integer_params:
                            cs_list.append((present := cs.IntegerUniformHyperparameter(f"{alg.name}_{i}_{key}",
                                                                                       param[0], param[1], parent > i)))
                        else:
                            cs_list.append((present := cs.UniformHyperparameter(f"{alg.name}_{i}_{key}", param[0],
                                                                                param[1], parent > i)))
                        cs_list.append(cs.CategoricalHyperparameter(f"{alg.name}_{i}_{key}", [None], ~present.cond))
        else:
            raise ValueError("Method must be 0, 1 or 2")
        self.config = cs.ConfigurationSpace(cs_list)
        self.name = "BOHB"
        self.cat_dependent_hyperparams = None

    def process_hparams(self, **kwargs):
        if self.method == generationMethod.FIXED:
            return self.process_fixed(**kwargs)
        elif self.method == generationMethod.BINARY:
            return self.process_binary(**kwargs)
        elif self.method == generationMethod.ALG_NUMBER:
            return self.process_alg_number(**kwargs, short_string=True)
        else:
            raise ValueError

    def score(self, config, seed=0):
        return -1*super().score(**config)

    def run(self, desc=None):
        opt = BOHB(self.config, self.score, max_budget=int(self.time_budget / 2), min_budget=1, n_proc=1)
        if not desc:
            desc = f"BOHB {[x().name for x in self.validation_functions]} {[x[0].name for x in self.algorithms]}"
        self.t = tqdm(total=self.time_budget, desc=desc,
                      unit="configuration", leave=False)
        # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = opt.optimize()
        res = result.best
        params = {x.name: x.value for x in res["hyperparameter"]}
        if len(params) > 0:
            params = self.process_hparams(**params)
            best_result = self.generate_clusterings(params)
            sim_mat, val_mat = self.evaluate_clusterings(best_result)
            return [x[1] for x in best_result], -1*res["loss"], sim_mat, val_mat, params
        else:
            return [], -1, _, _, {}
