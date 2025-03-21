import numpy as np

from dataset_fetcher import fetch_datasets

text = ""
text_syn = ""
if __name__ == '__main__':
    datasets = fetch_datasets()
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
    results = {}
    i = 0
    for dg in datasets:
        results[dg] = {"n": [], "d": [], "p": []}
        for name, dataset in datasets[dg].items():
            """n, dim = dataset["data"].shape
            clusters = len(np.unique(dataset["labels"]))
            results[dg]["n"].append(n)
            results[dg]["d"].append(dim)
            results[dg]["p"].append(clusters)
            continue"""
            if "ori" in name or name == "thy":
                continue
            n, dim = dataset["data"].shape
            clusters = len(np.unique(dataset["labels"]))
            arb, noi, ovl = "$\\times$", "$\\times$", "$\\times$"
            if name in arb_names: arb = "\\checkmark"
            if name in noi_names: noi = "\\checkmark"
            if name in ovl_names: ovl = "\\checkmark"
            outliers = (np.count_nonzero(dataset["labels"] == -1) / n)
            text += f" {name} & {dg} & {n} & {dim} & {clusters} & {outliers:.2f} & {arb} & {noi} & {ovl} \\\\\n"
            if "syn" in dg:
                text_syn += f" {name} & {n} & {dim} & {clusters} & {outliers:.2f} & {arb} & {noi} & {ovl} "
            if i % 2 == 0:
                text_syn += '&'
            else:
                text_syn += "\\\\ \\hline\n"
            i += 1
    # text = text.replace('_', '\_')
    # text = text.replace('syntetic', "syn")
    print(text_syn)
