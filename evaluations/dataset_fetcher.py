import csv
import os
import subprocess
import urllib.request
import zipfile

import numpy as np
import scipy.io
from sklearn import datasets
from ucimlrepo import fetch_ucirepo


def fetch_datasets_dbcv():
    ensure_datasets()
    datasets = {}
    base_directory = os.path.dirname(os.path.realpath(__file__))
    iris = fetch_ucirepo(id=53)
    labels = iris.data.targets.apply(lambda col: col.astype('category').cat.codes).to_numpy().reshape(-1)
    data = iris.data.features.to_numpy()
    datasets["iris"] = {"data": data, "labels": labels}
    wine = fetch_ucirepo(id=109)
    x = wine.data.features.to_numpy()
    y = wine.data.targets.to_numpy().reshape(-1)
    datasets["wine"] = {"data": x, "labels": y}
    ds = fetch_ucirepo(id=42)
    x = ds.data.features.to_numpy()
    y = ds.data.targets.to_numpy().reshape(-1)
    datasets["glass"] = {"data": x, "labels": y}
    with open(os.path.join(base_directory, "datasets", "synthetic_control.data")) as kdd_file:
        kdd_labels = []
        kdd_points = []
        idx = 0
        for row in csv.reader(kdd_file, delimiter=' '):
            kdd_labels.append(int(idx / 100))
            row = list(filter(lambda item: item != '', row))
            idx += 1
            if not row: continue
            a = list(map(float, row))
            kdd_points.append(a)
    datasets["kdd"] = {"data": kdd_points, "labels": kdd_labels}
    with open(os.path.join(base_directory, "datasets", "cell237.txt")) as cell237_file:
        cell237_labels = []
        cell237_points = []
        first = True
        for row in csv.reader(cell237_file, delimiter='\t'):
            if first:
                first = False
                continue
            if not row: continue
            cell237_labels.append(int(row[1]))
            a = list(map(float, row[2:18]))
            cell237_points.append(a)
    datasets["cell237"] = {"data": cell237_points, "labels": cell237_labels}
    with open(os.path.join(base_directory, "datasets", "cell384.txt")) as cell384_file:
        cell384_labels = []
        cell384_points = []
        first = True
        for row in csv.reader(cell384_file, delimiter='\t'):
            if first:
                first = False
                continue
            if not row: continue
            cell384_labels.append(int(row[1]))
            a = list(map(float, row[2:18]))
            cell384_points.append(a)
    datasets["cell384"] = {"data": cell384_points, "labels": cell384_labels}
    for name, dataset in datasets.items():
        data, labels = clean(dataset["data"], dataset["labels"])
        datasets[name] = {"data": data, "labels": labels}
    return datasets


def fetch_datasets_sklearn(n_samples=500):
    dataset = {}
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    dataset["circles"] = {"data": noisy_circles[0], "labels": noisy_circles[1]}
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    dataset["moons"] = {"data": noisy_moons[0], "labels": noisy_moons[1]}
    blobs = datasets.make_blobs(n_samples=n_samples, n_features=3)
    dataset["blobs"] = {"data": blobs[0], "labels": blobs[1]}
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    dataset["anisotropic"] = {"data": X_aniso, "labels": y}
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    dataset["varied"] = {"data": varied[0], "labels": varied[1]}
    return dataset


def fetch_datasets_cd_synthetic():
    import arff
    evaluations_directory = os.path.dirname(os.path.realpath(__file__))
    other_outlier = {"zelnik2": 2, "cure-t2-4k": 6}
    folder = os.path.join(evaluations_directory, "Clustering-Datasets", "02. Synthetic")
    datasets = {}
    for filename in os.listdir(folder):
        dataset_name = filename.split('.')[0]
        if dataset_name in datasets:
            continue
        if filename.endswith(".mat"):
            if f"{dataset_name}.mat" in os.listdir(folder):
                continue
            dataset = scipy.io.loadmat(os.path.join(folder, filename))
            data = dataset["data"]
            labels = dataset["label"]
        elif filename.endswith(".arff"):
            dataset = arff.load(os.path.join(folder, filename))
            data = []
            labels = []
            first = True
            fail = False
            try:
                for row in dataset:
                    if first:
                        try:
                            y = row["class"]
                        except KeyError:
                            try:
                                y = row["CLASS"]
                            except KeyError:
                                # print("Dataset "+dataset_name+" has no Attribute Class.")
                                fail = True
                                break
                        ls = list(row)
                        claspos = ls.index(y)
                        first = False
                    if not row:
                        continue
                    ls = list(row)
                    try:
                        labels.append(int(ls[claspos]))
                    except ValueError:
                        # print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Values.")
            if fail: continue
        else:
            continue
        if len(data) == 0:
            continue
        dataset, labels = clean(data, labels)
        if dataset_name in other_outlier:
            if labels.dtype == np.dtype('uint8'):
                labels = labels.astype(np.dtype('int8'))
            labels[labels == other_outlier[dataset_name]] = -1
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def fetch_datasets_cd_synthetic_split(arbitrary_shape=False, noise=False, overlap=False):
    datasets = fetch_datasets_cd_synthetic()
    all_names = list(datasets.keys())
    arb_names = ["2-cluster", "2sp2glob", "3-cluster", "2spiral", "3-spiral", "banana", "banana - Ori", "cluto-t4-8k",
                 "circle",
                 "cluto-t5-8k", "cluto-t7-10k", "cluto-t8-8k", "cure-t2-4k", "compound2", "compound",
                 "complex9", "complex8", "dartboard2", "dartboard1", "disk-1000n", "disk-3000n", "dense-disk-3000",
                 "dense-disk-5000", "disk-6000n", "disk-4600n", "disk-5000n", "disk-4500n", "disk-4000n", "donutcurves",
                 "ds3c3sc6", "dpb", "dpc", "donut3", "donut2", "flame", "impossible", "jain", "rings", "pearl",
                 "pathbased",
                 "smile2", "smile1", "smile3", "target", "spiralsquare", "spiral", "zelnik1", "zelnik3", "zelnik2",
                 "zelnik4",
                 "zelnik6", "zelink1", "zelink3"]
    noi_names = ["cluto-t4-8k", "cluto-t5-8k", "cluto-t7-10k", "cluto-t8-8k", "cure-t2-4k", "dpb", "dpc", "impossible",
                 "zelnik2", "zelnik4"]
    ovl_names = ["2d-20c-no0", "2dnormals", "3-cluster", "aggregation", "2d-3c-no123", "2d-4c-no9", "2d-10c", "circle",
                 "blobs", "disk-1000n", "diamond9", "D31", "cure-t2-4k1", "cure-t2-4k", "ds3c3sc6", "ds4c2sc8", "dpb",
                 "engytime", "elly-2d10c13s", "DS-577", "DS6", "elliptical_10_2", "sizes4", "sizes3", "sizes2", "S4",
                 "sizes1", "S3", "S2", "S1", "sizes5", "rings", "pathbased", "square2", "s-set1", "square1", "square5",
                 "square3", "square4", "spherical_5_2", "s-set2", "threenorm", "st900"]
    for jadb in [arb_names, noi_names, ovl_names]:
        for x in jadb:
            jada = datasets[x]
    if arbitrary_shape:
        all_names = list(set(all_names).intersection(set(arb_names)))
    if noise:
        all_names = list(set(all_names).intersection(set(noi_names)))
    if overlap:
        all_names = list(set(all_names).intersection(set(ovl_names)))
    return {key: value for key, value in datasets.items() if key in all_names}


def fetch_datasets_cd_uci():
    import arff
    evaluations_directory = os.path.dirname(os.path.realpath(__file__))
    if not evaluations_directory.endswith("evaluations"):
        evaluations_directory = os.path.join(evaluations_directory, "evaluations")
    folder = os.path.join(evaluations_directory, "Clustering-Datasets", "01. UCI")
    datasets = {}
    for filename in os.listdir(folder):
        dataset_name = filename.split('.')[0]
        if dataset_name in datasets:
            continue
        if filename.endswith(".mat"):
            dataset = scipy.io.loadmat(os.path.join(folder, filename))
            data = dataset["data"]
            labels = dataset["label"]
        elif filename.endswith(".arff"):
            if f"{dataset_name}.mat" in os.listdir(folder):
                continue
            dataset = arff.load(os.path.join(folder, filename))
            data = []
            labels = []
            first = True
            fail = False
            try:
                for row in dataset:
                    if first:
                        try:
                            y = row["class"]
                        except KeyError:
                            try:
                                y = row["CLASS"]
                            except KeyError:
                                try:
                                    y = row["Class"]
                                except KeyError:
                                    # print("Dataset "+dataset_name+" has no Attribute Class.")
                                    fail = True
                                    break
                        ls = list(row)
                        claspos = ls.index(y)
                        first = False
                    if not row:
                        continue
                    ls = list(row)
                    try:
                        labels.append(int(ls[claspos]))
                    except ValueError:
                        # print("Dataset "+dataset_name+" has Labels not direchtly castable "+str(ls[claspos]))
                        fail = True
                        break
                    data.append(ls[:claspos] + ls[claspos + 1:])
            except ValueError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Values.")
            except IndexError as Err:
                pass
                # print("Dataset " + dataset_name + " has bad Indizes.")
            data = np.array(data)
            labels = np.array(labels)
            if fail: continue
        else:
            continue
        if len(data) == 0:
            continue
        if data.dtype.type in [np.bytes_, np.str_]:
            data = data.astype(int)
        if np.isnan(data).any():
            data = data[~np.isnan(data).any(axis=1)]
        dataset, labels = clean(data, labels)
        datasets[dataset_name] = {"data": dataset, "labels": labels}
    return datasets


def dataset_names():
    names = {"dbcv": ["iris", "glass", "wine", "kdd", "cell237", "cell384"]}
    names["uci"] = list(fetch_datasets_cd_uci().keys())
    names["syntetic"] = list(fetch_datasets_cd_synthetic().keys())
    return names


def fetch_datasets():
    datasets = {"dbcv": fetch_datasets_dbcv()}
    datasets["uci"] = fetch_datasets_cd_uci()
    datasets["syntetic"] = fetch_datasets_cd_synthetic()
    # datasets["ac"] = fetch_datasets_ac()
    return datasets


def fetch_datasets_real_syn():
    return {"real": {**fetch_datasets_dbcv(), **fetch_datasets_cd_uci()}, "synthetic": fetch_datasets_cd_synthetic()}


def fetch_dataset(name: str):
    dss = fetch_datasets()
    for datasets in dss.values():
        if name in datasets:
            return datasets[name]
    raise ValueError("No dataset with " + name)


def clean(dataset, labels):
    dataset = np.array(dataset)
    if type(labels[0]) is list:
        true_labels = np.array([x for y in labels for x in y])
    elif type(labels) is list:
        true_labels = np.array(labels)
    else:
        true_labels = labels
    if type(true_labels) is np.ndarray:
        if len(true_labels.shape) == 2:
            if true_labels.shape[1] == 1:
                true_labels = true_labels.reshape(true_labels.shape[0])
            elif true_labels.shape[0] == 1:
                true_labels = true_labels.reshape(true_labels.shape[1])
            else:
                raise RuntimeWarning("Invalid Labels detected")
    valid_indizes = np.where(~np.isnan(true_labels))[0]
    nan_values = len(valid_indizes) != len(true_labels)
    if nan_values:
        true_labels = np.array([true_labels[x] for x in valid_indizes])
    dataset = dataset[valid_indizes]
    return dataset, true_labels


def clean_labels(labels):
    if type(labels[0]) is list:
        true_labels = np.array([x for y in labels for x in y])
    elif type(labels) is list:
        true_labels = np.array(labels)
    else:
        true_labels = labels
    if type(true_labels) is np.ndarray:
        if len(true_labels.shape) == 2:
            if true_labels.shape[1] == 1:
                true_labels = true_labels.reshape(true_labels.shape[0])
            elif true_labels.shape[0] == 1:
                true_labels = true_labels.reshape(true_labels.shape[1])
            else:
                raise RuntimeWarning("Invalid Labels detected")
    valid_indizes = np.where(~np.isnan(true_labels))[0]
    if len(valid_indizes) != len(true_labels):
        true_labels = np.array([true_labels[x] for x in valid_indizes])
    return true_labels


def ensure_datasets():
    base_directory = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(os.path.join(base_directory, "datasets")):
        os.makedirs(os.path.join(base_directory, "datasets"))
    files = {
        "cell237.txt": "http://faculty.washington.edu/kayee/cluster/logcho_237_4class.txt",
        "cell384.txt": "http://faculty.washington.edu/kayee/cluster/log_cellcycle_384_17.txt"
    }
    for file, url in files.items():
        file_name = os.path.join(base_directory, "datasets", file)
        if not os.path.exists(file_name):
            urllib.request.urlretrieve(url, file_name)
    if not os.path.exists(os.path.join(base_directory, "datasets", "synthetic_control.data")):
        zip_path = os.path.join(base_directory, "datasets", "synthetic_control.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/static/public/139/synthetic+control+chart+time+series.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract("synthetic_control.data", os.path.join(base_directory,"datasets"))
    if not os.path.isdir(os.path.join(base_directory, "Clustering-Datasets")):
        subprocess.run(["git", "clone", "https://github.com/milaan9/Clustering-Datasets",
                        os.path.join(base_directory, "Clustering-Datasets")])
