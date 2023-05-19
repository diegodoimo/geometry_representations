import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})
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

    intrinsic_dim = np.mean(ids, axis = 1)
    intrinsic_dim_err = np.std(ids, axis =1) / n_iter ** 0.5

    return intrinsic_dim, intrinsic_dim_err


def compute_id_from_dist(model_size, layers, distance_mat_folder, decimation = 32, n_iter=1):
    ids_twonn = []
    mus = []
    for l in layers:
        print(l)
        distances =  np.load(f'{distance_mat_folder}/{model_size}/imagenet/1000/imagenet_{model_size}_layer{l}_train_300cl_300im_dist.npy')
        dist_indices = np.load(f'{distance_mat_folder}/{model_size}/imagenet/1000/imagenet_{model_size}_layer{l}_train_300cl_300im_index.npy')
        d = data.Data(distances=(distances.astype(np.float64)[:, :], dist_indices[:, :]), maxk = 50, verbose = False)
        ids_twonn.append(d.compute_id_2NN(decimation = 1/decimation, fraction = 0.98, n_iter = n_iter))
        mus.append(d.intrinsic_dim_mus)

    intrinsic_dim_matrix = np.squeeze(np.vstack(ids_twonn))

    return intrinsic_dim_matrix[:, 0], intrinsic_dim_matrix[:, 1], mus




def compute_overlaps(model_size, layers, distance_mat_folder, labels, k):

    ov_gt = []
    for l in layers:
        dist_indices = np.load(f'{distance_mat_folder}/imagenet_train_nb_index_{model_size}_layer{l}_k10_cpt1000.npy')
        distances = np.zeros_like(dist_indices)
        d = data.Data(distances=(distances.astype(np.float64), dist_indices), maxk = 10, verbose = False)

        ov_gt.append(d.return_label_overlap(labels = labels, k = k))

    return ov_gt



#***********************************************************************************************************



parser = argparse.ArgumentParser()
parser.add_argument("--distance_mat_folder", type=str, default="./results/data")
parser.add_argument("--layers", type=int, nargs='+', default = np.arange(26))
parser.add_argument("--model_size", type=str, default = 'small')
parser.add_argument("--data_fraction", type=int, default = 32)
parser.add_argument("--n_iter", type=int, default = 4)
parser.add_argument("--label_path", type = str, default = './results/')
parser.add_argument("--compute_id_from_distance", action = 'store_true')
parser.add_argument("--plot_id", action = 'store_true')
parser.add_argument("--plot_overlap", action = 'store_true')
parser.add_argument("--save_mus", action = 'store_true')
args = parser.parse_args()

#*****************************

if args.plot_id:

    model_size = 'small'
    mus =  np.load(f'{args.distance_mat_folder}/imagenet_train_mus_{model_size}_dec32_cpt1000_rep4.npy')
    ids_small, err_small = compute_id_from_mus(mus)

    model_size = 'medium'
    mus =  np.load(f'{args.distance_mat_folder}/imagenet_train_mus_{model_size}_dec32_cpt1000_rep4.npy')
    ids_medium, err_medium = compute_id_from_mus(mus)

    model_size = 'large'
    mus =  np.load(f'{args.distance_mat_folder}/imagenet_train_mus_{model_size}_dec32_cpt1000_rep4.npy')
    ids_large, err_large = compute_id_from_mus(mus)

    fig = plt.figure(figsize = (4, 3))
    ax = fig.add_subplot()
    ax.errorbar(np.arange(len(ids_small))/len(ids_small), ids_small, 3*err_small, label = 'small')
    ax.errorbar(np.arange(len(ids_medium))/len(ids_medium), ids_medium, 3*err_medium, label = 'medium')
    ax.errorbar(np.arange(len(ids_large))/len(ids_large), ids_large, 3*err_large, label = 'large')
    ax.legend()
    ax.set_ylabel('ID', fontsize = 13)
    ax.set_xlabel('relative depth', fontsize = 12)
    ax.set_title('Intrinsic dimension')
    plt.tight_layout()
    plt.savefig('./results/intrinsic_dimension.png', dpi = 150)



if args.plot_overlap:
    layer_dict = {'small': np.array([0, 6, 12, 18, 24]),
                  'medium': np.array([0, 6, 12, 18, 24, 30, 36]),
                  'large': np.array([0, 8, 16, 24, 32, 40, 48])
    }

    labels = np.load(f'{args.distance_mat_folder}/imagenet_train_labels.npy')
    ov_gt_large = compute_overlaps('large', layer_dict['large'], args.distance_mat_folder, labels=labels, k = 10)
    ov_gt_medium = compute_overlaps('medium', layer_dict['medium'], args.distance_mat_folder, labels=labels, k = 10)
    ov_gt_small = compute_overlaps('small', layer_dict['small'], args.distance_mat_folder, labels=labels, k = 10)

    
    fig = plt.figure(figsize = (4, 3))
    ax = fig.add_subplot()
    sns.lineplot(x = layer_dict['small']/layer_dict['small'][-1], y = ov_gt_small, marker = 'o', label = 'small')
    sns.lineplot(x = layer_dict['medium']/layer_dict['medium'][-1], y = ov_gt_medium, marker = 'o', label = 'medium')
    sns.lineplot(x = layer_dict['large']/layer_dict['large'][-1], y = ov_gt_large, marker = 'o', label = 'large')
    ax.legend()
    ax.set_ylabel('$\chi^{l, gt}$', fontsize = 14)
    ax.set_xlabel('relative depth', fontsize = 12)
    ax.set_title('Neighborhood consistency w.r.t classification labels')
    plt.tight_layout()
    plt.savefig('./results/overlap_ground_truth.png', dpi = 150)


if args.compute_id_from_distance:

    ids, err, mus = compute_id_from_dist(args.model_size, args.layers, args.distance_mat_folder, decimation = 32, n_iter = args.n_iter)

    plt.errorbar(np.arange(len(ids))/len(ids), ids, 3*err)

