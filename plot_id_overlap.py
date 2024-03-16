import argparse
import numpy as np
import matplotlib.pyplot as plt
from id_overlap_helpers import (
    compute_id_from_mus,
    compute_overlaps,
    compute_id_from_dist,
)

from matplotlib.gridspec import GridSpec
import seaborn as sns
import pathlib


# ***********************************************************************************************************


parser = argparse.ArgumentParser()
parser.add_argument("--distance_mat_folder", type=str, default="./results/data")
parser.add_argument("--target_dir", type=str, default="./results/plots")
parser.add_argument("--compute_id_from_distances", action="store_true")
args = parser.parse_args()


pathlib.Path(args.target_dir).mkdir(parents=True, exist_ok=True)


# *****************************
# intrinsic dimension calculation

intrinsic_dimension = {}
for model_size in ["small", "medium", "large"]:

    mus = np.load(
        f"{args.distance_mat_folder}/imagenet_train_mus_{model_size}_dec32_cpt1000_rep4.npy"
    )
    ids, _ = compute_id_from_mus(mus)
    intrinsic_dimension["igpt-" + model_size] = ids

# for esm2 model the intrinsic dimension in alredy computed
intrinsic_dimension["esm-35M"] = np.load(f"{args.distance_mat_folder}/id-esm2-35M.npy")
intrinsic_dimension["esm-650M"] = np.load(
    f"{args.distance_mat_folder}/id-esm2-650M.npy"
)
intrinsic_dimension["esm-3B"] = np.load(f"{args.distance_mat_folder}/id-esm2-3B.npy")


# ****************************
# overlap with classes computation

overlap = {}
labels = np.load(f"{args.distance_mat_folder}/imagenet_train_labels.npy")
layer_dict = {
    "small": np.array([0, 6, 12, 18, 24]),
    "medium": np.array([0, 6, 12, 18, 24, 30, 36]),
    "large": np.array([0, 8, 16, 24, 32, 40, 48]),
}

for model_size in ["small", "medium", "large"]:
    overlap["igpt-" + model_size] = compute_overlaps(
        model_size,
        layer_dict[model_size],
        args.distance_mat_folder,
        labels=labels,
        k=10,
    )

overlap["esm-35M"] = np.load(f"{args.distance_mat_folder}/no-gt-esm2-35M.npy")
overlap["esm-650M"] = np.load(f"{args.distance_mat_folder}/no-gt-esm2-650M.npy")
overlap["esm-3B"] = np.load(f"{args.distance_mat_folder}/no-gt-esm2-3BM.npy")


# ******************************************************************
# *******************************************************************

sns.set_style("whitegrid")

colors = {"igpt-small": "#FFDB00", "igpt-medium": "#ee7b06", "igpt-large": "#a12424"}


fig = plt.figure(figsize=(4.5, 6.5))
gs = GridSpec(2, 1)
ax = fig.add_subplot(gs[0])
for model in ["igpt-small", "igpt-medium", "igpt-large"]:
    ids = intrinsic_dimension[model]
    layers = np.arange(len(ids)) / len(ids)
    sns.lineplot(x=layers, y=ids, color=colors[model], marker=".", label=model)
ax.set_ylabel("intrinsic dimension", fontsize=12)


ax = fig.add_subplot(gs[1])
for model in ["igpt-small", "igpt-medium", "igpt-large"]:
    ovs = overlap[model]
    layers = np.arange(len(ovs)) / len(ovs)
    sns.lineplot(x=layers, y=ovs, color=colors[model], marker="o", label=model)
ax.set_xlabel("relative depth", fontsize=12)
ax.set_ylabel("overlap labels", fontsize=12)
ax.set_ylim(-0.01, 0.4)

gs.tight_layout(fig)
plt.savefig(f"{args.target_dir}/igpt_id_ov_labels.png", dpi=150)


# ***************************************************************************

sns.set_style("whitegrid")


colors = {
    "esm-35M": "#7AD151",
    "esm-650M": (42 / 235, 120 / 235, 142 / 235),
    "esm-3B": "#414487",
}


fig = plt.figure(figsize=(4.5, 6.5))
gs = GridSpec(2, 1)
ax = fig.add_subplot(gs[0])
for model in ["esm-35M", "esm-650M", "esm-3B"]:
    ids = intrinsic_dimension[model]
    layers = np.arange(len(ids)) / len(ids)
    sns.lineplot(x=layers, y=ids, color=colors[model], marker=".", label=model)
ax.set_ylabel("intrinsic dimension", fontsize=12)


ax = fig.add_subplot(gs[1])
for model in ["esm-35M", "esm-650M", "esm-3B"]:
    ovs = overlap[model]
    layers = np.arange(len(ovs)) / len(ovs)
    sns.lineplot(x=layers, y=ovs, color=colors[model], marker=".", label=model)
ax.set_xlabel("relative depth", fontsize=12)
ax.set_ylabel("overlap labels", fontsize=12)
ax.set_ylim(0, 1)

gs.tight_layout(fig)
plt.savefig(f"{args.target_dir}/esm_id_ov_labels.png", dpi=150)


if args.compute_id_from_distance:
    ids, err, mus = compute_id_from_dist(
        args.model_size,
        args.layers,
        args.distance_mat_folder,
        decimation=32,
        n_iter=args.n_iter,
    )
    plt.errorbar(np.arange(len(ids)) / len(ids), ids, 3 * err)
