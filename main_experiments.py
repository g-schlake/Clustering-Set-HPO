from auxiliaries.misc import generationMethod
from measures.registry import get_measures
from algorithms.registry import get_algorithms
from evaluations.dataset_fetcher import fetch_datasets_real_syn
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from interestingness_optimizer.interestingness_functions import naive_interestingness_avg, \
    naive_interestingness_mean_sim, \
    naive_interestingness_min_sim
from interestingness_optimizer.interestingness_scorer_opt import InterestingnessScorerOptunity
from interestingness_optimizer.interestingness_scorer_bohb import InterestingnessScorerBOHB
import os
import time
import platform

if __name__ == '__main__':

    if platform.system() == "Linux":
        from interestingness_optimizer.interestingness_scorer_smac import InterestingnessScorerSMAC_AC, \
            InterestingnessScorerSMAC_BB, InterestingnessScorerSMAC_HPO

        solvers = [InterestingnessScorerSMAC_HPO, InterestingnessScorerSMAC_BB, InterestingnessScorerSMAC_AC,
                   InterestingnessScorerBOHB, InterestingnessScorerOptunity]
    else:
        solvers = [InterestingnessScorerBOHB, InterestingnessScorerOptunity]
    validation_functions = get_measures()[:2]
    algs = [(x, 5) for x in get_algorithms()[:3]]
    dgs = fetch_datasets_real_syn()
    df_res = pd.DataFrame(columns=["Dataset", "Interestingness", "HPO", "Method", "Score", "Hyperparams", "Duration"])
    i_funcs = {"naive_max": naive_interestingness_avg, "naive_min": naive_interestingness_min_sim,
               "naive_avg": naive_interestingness_mean_sim}
    for dg in dgs:
        dss = dgs[dg]
        for ds in dss:
            data, labels = dss[ds]["data"], dss[ds]["labels"]
            for solver in solvers:
                if os.path.exists(f"output/{ds}_{solver.__name__}.csv"):
                    ds_res = pd.read_csv(f"output/{ds}_{solver.__name__}.csv", index_col=0)
                    df_res = pd.concat([df_res, ds_res])
                    continue
                ds_res = pd.DataFrame(
                    columns=["Dataset", "Interestingness", "HPO", "Method", "Score", "Hyperparams", "Duration"])
                for interest in ["naive_avg"]:
                    for method in list(generationMethod):
                        start = time.time()
                        scorer = solver(validation_functions, algs, data, adjusted_rand_score, i_funcs[interest],
                                        method=method, time_budget=100)
                        res, score, _, validations, results = scorer.run(
                            desc=f"{ds} {scorer.name} {interest} {method.name}")
                        end = time.time()
                        print(f"Result for {ds} {scorer.name} {interest} {method.name} is {score}")
                        df_res.loc[len(df_res.index)] = [ds, interest, scorer.name, method.name, score, str(results),
                                                         end - start]
                        ds_res.loc[len(ds_res.index)] = [ds, interest, scorer.name, method.name, score, str(results),
                                                         end - start]
                ds_res.to_csv(f"output/{ds}_{solver.__name__}.csv")
        df_res.to_csv(f"interestingness_hpo_{dg}.csv")
