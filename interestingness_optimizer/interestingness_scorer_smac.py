import warnings

from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from ConfigSpace.conditions import EqualsCondition, GreaterThanCondition
from smac import Scenario, AlgorithmConfigurationFacade, BlackBoxFacade, HyperparameterOptimizationFacade
from tqdm import tqdm

from auxiliaries.misc import generationMethod
from interestingness_optimizer.interestingness_scorer import InterestingnessScorer


class InterestingnessScorerSMACBase(InterestingnessScorer):

    def __init__(self, validation_function, algorithms, data, similarity_function, interestingness_function, method=1,
                 time_budget=20):
        super().__init__(validation_function, algorithms, data, similarity_function, interestingness_function, method)
        self.time_budget = time_budget
        self.config = ConfigurationSpace()
        params = {}
        conditions = []
        if method == generationMethod.FIXED:
            self.slots = []
            for alg, number in algorithms:
                for i in range(number):
                    for key, value in self.algorithm_dict[alg.name][0].hyperparams.items():
                        param_name = f"{alg.name}_{i}_{key}"
                        if key in alg.integer_params:
                            params[param_name] = Integer(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                        else:
                            params[param_name] = Float(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                    self.slots.append(f"{alg.name}_{i}")
        elif method == generationMethod.BINARY:
            for alg, number in algorithms:
                for i in range(number):
                    params[f"{alg.name}_{i}"] = Categorical(f"{alg.name}_{i}", [True, False])
                    for key, value in self.algorithm_dict[alg.name][0].hyperparams.items():
                        param_name = f"{alg.name}_{i}_{key}"
                        if key in alg.integer_params:
                            params[param_name] = Integer(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                        else:
                            params[param_name] = Float(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                    conditions.append(
                        EqualsCondition(child=params[param_name], parent=params[f"{alg.name}_{i}"], value=True))

        elif method == generationMethod.ALG_NUMBER:
            for alg, number in algorithms:
                params[alg.name] = Integer(alg.name, (0, number))
                for i in range(number):
                    for key, value in self.algorithm_dict[alg.name][0].hyperparams.items():
                        param_name = f"{alg.name}_{i}_{key}"
                        if key in alg.integer_params:
                            params[param_name] = Integer(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                        else:
                            params[param_name] = Float(param_name, (alg.hyperparams[key][0], alg.hyperparams[key][1]))
                        conditions.append(
                            GreaterThanCondition(child=params[param_name], parent=params[alg.name], value=i))
        self.config.add(params.values())
        if len(conditions) > 0:
            self.config.add(conditions)

    def process_hparams(self, **kwargs):
        if self.method == generationMethod.FIXED:
            return self.process_fixed(**kwargs, to_int=False)
        elif self.method == generationMethod.BINARY:
            return self.process_binary(**kwargs, to_int=False)
        elif self.method == generationMethod.ALG_NUMBER:
            return self.process_alg_number(to_int=False, short_string=True, **kwargs)

    def score(self, config, seed=0):
        return -1 * super().score(**config)

    def run(self, desc=None):
        if not desc:
            desc = f"{self.name} {[x().name for x in self.validation_functions]} {[x[0].name for x in self.algorithms]}"
        scenario = Scenario(self.config, n_trials=1 * self.time_budget, seed=-1)
        smac = self.facade(scenario, self.score, overwrite=True, logging_level=40)
        self.t = tqdm(total=self.time_budget, desc=desc,
                      unit="configuration", leave=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = smac.optimize()
        params = self.process_hparams(**res)
        best_result = self.generate_clusterings(params)
        sim_mat, val_mat = self.evaluate_clusterings(best_result)
        score = self.interestingness_function(sim_mat, val_mat)
        return [x[1] for x in best_result], score, sim_mat, val_mat, res


class InterestingnessScorerSMAC_HPO(InterestingnessScorerSMACBase):
    def __init__(self, validation_function, algorithms, data, similarity_function, interestingness_function, method=1,
                 time_budget=20):
        super().__init__(validation_function, algorithms, data, similarity_function, interestingness_function, method,
                         time_budget)
        self.facade = HyperparameterOptimizationFacade
        self.name = "SMAC HPO"


class InterestingnessScorerSMAC_BB(InterestingnessScorerSMACBase):
    def __init__(self, validation_function, algorithms, data, similarity_function, interestingness_function, method=1,
                 time_budget=20):
        super().__init__(validation_function, algorithms, data, similarity_function, interestingness_function, method,
                         time_budget)
        self.facade = BlackBoxFacade
        self.name = "SMAC BB"


class InterestingnessScorerSMAC_AC(InterestingnessScorerSMACBase):
    def __init__(self, validation_function, algorithms, data, similarity_function, interestingness_function, method=1,
                 time_budget=20):
        super().__init__(validation_function, algorithms, data, similarity_function, interestingness_function, method,
                         time_budget)
        self.facade = AlgorithmConfigurationFacade
        self.name = "SMAC AC"
