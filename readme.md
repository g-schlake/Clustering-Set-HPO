# Clustering Set HPO
This GitHub Repository contains the experiments to explore different search spaces for Automated Exploratory Clustering. 
It should help to reproduce the results in 

> Anyonymous Authors
> 
> "Search Spaces for Hyperparameter Optimization of Interesting Clustering Sets" 
> 
> hopefully AutoML 2025

If you use this repo for your work, please cite the aforementioned paper.

## Setup

The Repo was tested with Python 3.10 and Linux. On Windows, SMAC does not work and is not used.
As of Mar. 2024, Python versions > 3.10 do not work as the required library `pyrfr` hasn't been ported  to current Python versions yet.

On Linux, `$ bash setup_env.bash` creates a Python virtual environment in the repo and installs all necessary dependencies from the `requirements.txt`.
A complete setup and reproduction of all experiments should work like this:

```bash
bash setup_env.bash
source .venv/bin/activate
python main_experiments.py
```

##  Datasets

The `dataset_fetcher` in the `auxiliaries` folder retrieves  the 
synthetic and real datasets from the [clustering datasets repository](https://github.com/milaan9/Clustering-Datasets) and
additionally, the datasets [`cell237`](http://faculty.washington.edu/kayee/cluster/logcho_237_4class.txt), 
[`cell384`](http://faculty.washington.edu/kayee/cluster/log_cellcycle_384_17.txt), 
[`glass`](https://archive.ics.uci.edu/dataset/42/glass+identification), 
[`iris`](https://archive.ics.uci.edu/dataset/53/iris), 
[`kdd`](https://archive.ics.uci.edu/dataset/139/synthetic+control+chart+time+series) and 
[`wine`](https://archive.ics.uci.edu/dataset/186/wine+quality) are put in
the `datasets` folder in `evaluations` to also use these datasets.
We adopted some of the datasets from the gitHub repo slightly to make them compatible with our program.

## Experiments
Our experiments can be done using the `main_experiments`.
This will automatically retrieve all necessary datasets and recompute the results of our experiments.
The output will be stored in the `output` folder. 
If you want a minimum working example on a small dataset (like `iris`), you can skip the computation of all other 
datasets  by inserting a line like ``if ds != "iris": continue`` as line 38 in `main_experiments`.

## Reported results
All results used for our paper can be found under `reported_results/interestingness_hpo.csv`. 

## Graphic reproducability
The graphics and table of the `Result` part of our paper can mostly be reproduced using 
the notebook `graphics_reproducability`.
