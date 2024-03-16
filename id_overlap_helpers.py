import numpy as np
from scipy.optimize import curve_fit
from dadapy import data


def _compute_id_2NN(mus, fraction=0.98, algorithm="base"):

    N = mus.shape[0]
    N_eff = int(N * fraction)
    log_mus = np.log(mus)
    log_mus_reduced = np.sort(log_mus)[:N_eff]

    if algorithm == "ml":
        intrinsic_dim = (N - 1) / np.sum(log_mus)

    elif algorithm == "base":
        y = -np.log(1 - np.arange(1, N_eff + 1) / N)

        def func(x, m):
            return m * x

        intrinsic_dim, _ = curve_fit(func, log_mus_reduced, y)

    else:
        raise ValueError("Please select a valid algorithm type")

    return intrinsic_dim


def compute_id_from_mus(mus):

    n_layers = mus.shape[0]
    n_iter = mus.shape[1]
    ids = np.zeros((n_layers, n_iter))

    for i in range(n_layers):
        for j in range(n_iter):
            ids[i, j] = _compute_id_2NN(mus[i, j])

    intrinsic_dim = np.mean(ids, axis=1)
    intrinsic_dim_err = np.std(ids, axis=1) / n_iter**0.5

    return intrinsic_dim, intrinsic_dim_err


def compute_id_from_dist(
    model_size, layers, distance_mat_folder, decimation=32, n_iter=1
):
    ids_twonn = []
    mus = []
    for l in layers:
        print(l)
        distances = np.load(
            f"{distance_mat_folder}/{model_size}/imagenet/1000/imagenet_{model_size}_layer{l}_train_300cl_300im_dist.npy"
        )
        dist_indices = np.load(
            f"{distance_mat_folder}/{model_size}/imagenet/1000/imagenet_{model_size}_layer{l}_train_300cl_300im_index.npy"
        )
        d = data.Data(
            distances=(distances.astype(np.float64)[:, :], dist_indices[:, :]),
            maxk=50,
            verbose=False,
        )
        ids_twonn.append(
            d.compute_id_2NN(decimation=1 / decimation, fraction=0.98, n_iter=n_iter)
        )
        mus.append(d.intrinsic_dim_mus)

    intrinsic_dim_matrix = np.squeeze(np.vstack(ids_twonn))

    return intrinsic_dim_matrix[:, 0], intrinsic_dim_matrix[:, 1], mus


def compute_overlaps(model_size, layers, distance_mat_folder, labels, k):

    ov_gt = []
    for l in layers:
        dist_indices = np.load(
            f"{distance_mat_folder}/imagenet_train_nb_index_{model_size}_layer{l}_k10_cpt1000.npy"
        )
        distances = np.zeros_like(dist_indices)
        d = data.Data(
            distances=(distances.astype(np.float64), dist_indices),
            maxk=10,
            verbose=False,
        )
        ov_gt.append(d.return_label_overlap(labels=labels, k=k))

    return ov_gt
