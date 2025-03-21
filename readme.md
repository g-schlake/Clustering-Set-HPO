# Clustering Set HPO
This GitHub Repository contains the experiments to explore different search spaces for Automated Exploratory Clustering.
This `main_experiments.py` evaluates the experiments in 

> Anyonymous Authors
> 
> "Search Spaces for Hyperparameter Optimization of Interesting Clustering Sets" 
> 
> hopefully AutoML 2025

The `dataset_fetcher` in the `auxiliaries` folder retrieves  the 
synthetic and real datasets from the [clustering datasets repository](https://github.com/milaan9/Clustering-Datasets).
The repository needs to be cloned in the `evaluations` folder.
Additionally, the datasets [`cell237`](http://faculty.washington.edu/kayee/cluster/logcho_237_4class.txt), 
[`cell384`](http://faculty.washington.edu/kayee/cluster/log_cellcycle_384_17.txt), 
[`glass`](https://archive.ics.uci.edu/dataset/42/glass+identification), 
[`iris`](https://archive.ics.uci.edu/dataset/53/iris), 
[`kdd`](https://archive.ics.uci.edu/dataset/139/synthetic+control+chart+time+series) and 
[`wine`](https://archive.ics.uci.edu/dataset/186/wine+quality) can be put in
the `datasets` folder in `evaluations` to also use these datasets.
We adopted some of these datasets slightly to make them compatible with our program. 
The Repo was tested with Python 3.10.
