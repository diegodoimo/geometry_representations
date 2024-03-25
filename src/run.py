import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from imageio import imwrite
from scipy.special import softmax
from tensorflow.contrib.training import HParams
from tqdm import tqdm

from model import model
from utils import iter_data, count_parameters


# *** ADDED
from collections import defaultdict
from pairwise_distance import compute_distances
import pathlib


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data and I/O
    # WE REQUIRE THE CKPT DIR and DATA DIR INSTEAD (SEE BELOW)
    # parser.add_argument("--data_path", type=str, default="/root/downloads/imagenet")
    # parser.add_argument(
    #     "--ckpt_path", type=str, default="/root/downloads/model.ckpt-1000000"
    # )
    parser.add_argument(
        "--color_cluster_path", type=str, default="/root/downloads/kmeans_centers.npy"
    )
    # WE REPLACE IS WITH 'results_dir' (SEE BELOW)
    # parser.add_argument("--save_dir", type=str, default="/root/save/")

    # WE DEFINE THE PARAMETER INSIDE THE MAIN SCRIPT
    # model
    # parser.add_argument("--n_embd", type=int, default=512)
    # parser.add_argument("--n_head", type=int, default=8)
    # parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument(
        "--n_px", type=int, default=32, help="image height or width in pixels"
    )
    parser.add_argument(
        "--n_vocab", type=int, default=512, help="possible values for each pixel"
    )

    parser.add_argument(
        "--bert",
        action="store_true",
        help="use the bert objective (defaut: autoregressive)",
    )
    parser.add_argument("--bert_mask_prob", type=float, default=0.15)
    parser.add_argument(
        "--clf", action="store_true", help="add a learnable classification head"
    )

    # parallelism
    parser.add_argument("--n_sub_batch", type=int, default=8, help="per-gpu batch size")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to distribute training across",
    )

    # mode
    # NOT USED. BY DEFAULT WE EXTRACT REPRESENTATIONS
    # parser.add_argument(
    #     "--eval",
    #     action="store_true",
    #     help="evaluates the model, requires a checkpoint and dataset",
    # )
    # parser.add_argument(
    #     "--sample",
    #     action="store_true",
    #     help="samples from the model, requires a checkpoint and clusters",
    # )

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="seed for random, np, tf")

    # ***OUR ARGUMENTS***
    parser.add_argument("--model", type=str, choices=["s", "m", "l"], default="s")
    parser.add_argument("--data_dir", type=str, default=".data/imagenet")
    parser.add_argument("--ckpt_dir", type=str, default=".models/igpt")
    parser.add_argument("--ckpt_iter", type=int, default=1000000)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--valset", action="store_true")
    parser.add_argument("--activations", action="store_true")
    parser.add_argument("--save_labels", action="store_true")
    parser.add_argument("--not_trained", action="store_true")
    parser.add_argument("--maxk", type=int, default=50)
    parser.add_argument("--nimg_cat", type=int, default=300)
    parser.add_argument("--working_memory", type=int, default=1024)

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# ***LOADING DATA FUNCTION: modified to extract one dataset at a time***
def load_data(args):

    data_path = args.data_dir + "/imagenet"

    if args.valset:
        Xdata = np.load(f"{data_path}_vaX.npy")
        Ydata = np.load(f"{data_path}_vaY.npy")
        description = "valid"

    elif args.testset:
        Xdata = np.load(f"{data_path}_teX.npy")
        Ydata = np.load(f"{data_path}_teY.npy")
        description = "test"
    else:
        Xdata = np.load(f"{data_path}_trX.npy")
        Ydata = np.load(f"{data_path}_trY.npy")
        description = "train"

    # BY DEFAULT WE SAMPLE 90K IMAGES FROM THE TRAINING SET.
    # IF YOU WANT TO EXAMINE THE TEST /VALID SET OR EXTRACT THE FULL TRAINING SET
    # COMMENT THE LINES BELOW

    # SELECT 300 CLASSES
    labels = np.load("./hier_nucl_labels.npy")
    full_targets = np.where(Ydata)[1]  # get real numbers from one hot
    subset_indices = []  # indices new subset
    # SELECT 300 IMAGES PER CLASS
    for i in range(len(labels)):
        subset_indices.extend(np.where(full_targets == labels[i])[0][: args.nimg_cat])
    subset_indices = np.array(subset_indices)

    print("initial training shapes", Xdata.shape, Ydata.shape)
    Xdata = Xdata[subset_indices]
    Ydata = Ydata[subset_indices]
    print("training shapes after subsampling", Xdata.shape, Ydata.shape)

    return Xdata, Ydata, description


# ***FILLER FUNCTION ADDED***
def fill_data(Xdata, Ydata, args):
    n_batch = args.n_sub_batch * args.n_gpu
    nfiller = (n_batch - Xdata.shape[0] % n_batch) % n_batch

    Xfiller = np.ones((nfiller, Xdata.shape[1]), dtype=Xdata.dtype)
    Yfiller = np.zeros((nfiller, Ydata.shape[1]), dtype=Ydata.dtype)

    Xdata = np.vstack((Xdata, Xfiller))
    Ydata = np.vstack((Ydata, Yfiller))
    return Xdata, Ydata


def set_hparams(args):
    # ***ADDED layers_to_extract_argument
    return HParams(
        n_ctx=args.n_px * args.n_px,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_vocab=args.n_vocab,
        bert=args.bert,
        bert_mask_prob=args.bert_mask_prob,
        clf=args.clf,
        layers_to_extract=args.layers_to_extract,
    )


def create_model(x, y, n_gpu, hparams):
    gen_logits = []
    gen_loss = []
    clf_loss = []
    tot_loss = []
    accuracy = []
    # ***ACTIVATION LIST ADDED***
    activations = []

    trainable_params = None
    for i in range(n_gpu):
        with tf.device("/gpu:%d" % i):
            results = model(hparams, x[i], y[i], reuse=(i != 0))

            gen_logits.append(results["gen_logits"])
            gen_loss.append(results["gen_loss"])
            clf_loss.append(results["clf_loss"])

            if hparams.clf:
                tot_loss.append(results["gen_loss"] + results["clf_loss"])
            else:
                tot_loss.append(results["gen_loss"])
            # ***ADDED
            accuracy.append(results["accuracy"])

            if i == 0:
                trainable_params = tf.trainable_variables()
                print("trainable parameters:", count_parameters())

    return (
        trainable_params,
        gen_logits,
        gen_loss,
        clf_loss,
        tot_loss,
        accuracy,
        activations,
    )


def reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, n_gpu):
    with tf.device("/gpu:0"):
        for i in range(1, n_gpu):
            gen_loss[0] += gen_loss[i]
            clf_loss[0] += clf_loss[i]
            tot_loss[0] += tot_loss[i]
            accuracy[0] += accuracy[i]
        gen_loss[0] /= n_gpu
        clf_loss[0] /= n_gpu
        tot_loss[0] /= n_gpu
        accuracy[0] /= n_gpu


def save_activations(act_dict, dir_path, nsamples, description, args):

    dir_path += "/activations"
    if not os.path.isdir(f"{dir_path}"):
        pathlib.Path(f"{dir_path}").mkdir(parents=True, exist_ok=True)

    for lay, act in act_dict.items():
        act = np.vstack(act)[:nsamples]
        if len(act.shape) > 2:  # some axis equal 1
            act = np.squeeze(act)  # Nx512
        np.save(
            f"{dir_path}/imagenet_{args.model_size}_layer{lay}_{description}_act.npy",
            act,
        )


def save_distances(act_dict, dir_path, nsamples, description, args):
    dir_path += "/distance_matrices"
    if not os.path.isdir(f"{dir_path}"):
        pathlib.Path(f"{dir_path}").mkdir(parents=True, exist_ok=True)

    for lay, act in act_dict.items():
        act = np.vstack(act)[:nsamples]
        if len(act.shape) > 2:  # some axis equal 1
            act = np.squeeze(act)  # Nx512

        distances, dist_index, mus, _ = compute_distances(
            X=act,
            n_neighbors=args.maxk + 1,
            n_jobs=1,
            working_memory=args.working_memory,
            range_scaling=1050,
            argsort=False,
        )

        np.save(
            f"{dir_path}/imagenet_{args.model_size}_layer{lay}_{description}_dist.npy",
            distances,
        )
        np.save(
            f"{dir_path}/imagenet_{args.model_size}_layer{lay}_{description}_index.npy",
            dist_index,
        )
        np.save(
            f"{dir_path}/imagenet_{args.model_size}_layer{lay}_{description}_mus.npy",
            mus,
        )


def evaluate(
    sess,
    evX,
    evY,
    X,
    Y,
    gen_loss,
    clf_loss,
    accuracy,
    n_batch,
    activations,
    nsamples,
    args,
    desc,
):
    metrics = []

    # *** ADDED
    act_dict = defaultdict(list)
    n = len(evX)
    nbatches = n // n_batch
    print(f"nsamples: {nsamples}")
    print(f"n images: {n}")
    print(f"batch_size: {n_batch}")
    print(f"n batches: {nbatches}")
    sys.stdout.flush()

    final_act = {}
    for i, name in enumerate(args.layers_to_extract):
        final_act[name] = np.zeros((nsamples, args.embdim[name]))

    print("initialized activation dictionaries")
    sys.stdout.flush()

    # ***WE MODIFIED THE LOGIC OF THE LOOP BELOW TO EXTRACT THE ACTIVATIONS
    for k, (xmb, ymb) in enumerate(
        iter_data(evX, evY, n_batch=n_batch, truncate=True, verbose=True)
    ):
        if (k + 1) % 1 == 0:
            print(f"{i*n_batch/1000}k/{nsamples}")
            sys.stdout.flush()

        (
            eval_gen_loss_tmp,
            eval_clf_loss_tmp,
            eval_accuracy_tmp,
            act_dict_tmp,
        ) = sess.run(
            [gen_loss[0], clf_loss[0], accuracy[0], activations[0]],
            {X: xmb, Y: ymb},
        )

        metrics.append((eval_gen_loss_tmp, eval_clf_loss_tmp, eval_accuracy_tmp))
        # ***STORE THE EXTRACETD BATCH IN THE FINAL DICTIONARY
        for lay, act in act_dict_tmp.items():
            if k == nbatches:
                act_dict[lay][k * n_batch :] = act
            else:
                act_dict[lay][k * n_batch : (k + 1) * n_batch] = act

    eval_gen_loss, eval_clf_loss, eval_accuracy = [np.mean(m) for m in zip(*metrics)]
    print(
        f"{desc} gen: {eval_gen_loss:.4f} clf: {eval_clf_loss:.4f} acc: {eval_accuracy:.2f}"
    )

    dir_path = f"{args.results_dir}/{args.model_size}"

    # ***SAVING THE REPRESENTATIONS
    if args.representations:
        save_activations(act_dict, dir_path, nsamples, desc, args)
    else:
        save_distances(act_dict, dir_path, nsamples, desc, args)

    labels = np.where(evY)[1]
    np.save(f"{dir_path}/imagenet_labels.npy", labels[:nsamples])


def main(args):
    set_seed(args.seed)

    n_batch = args.n_sub_batch * args.n_gpu
    n_class = 1000  # we just use imagenet

    X = tf.placeholder(tf.int32, [n_batch, args.n_px * args.n_px])
    Y = tf.placeholder(tf.float32, [n_batch, n_class])

    x = tf.split(X, args.n_gpu, 0)
    y = tf.split(Y, args.n_gpu, 0)

    # ***MODEL PARAMS
    if args.model == "s":
        args.n_embd = 512
        args.n_head = 8
        args.n_layer = 24
        args.model_size = "small"
        args.ckpt_dir += f"/model.ckpt-{args.ckpt_iter}"

    elif args.model == "m":
        args.n_embd = 1024
        args.n_head = 8
        args.n_layer = 36
        args.model_size = "medium"
        args.ckpt_dir += f"/model.ckpt-{args.ckpt_iter}"

    elif args.model == "l":
        args.n_embd = 1536
        args.n_head = 16
        args.n_layer = 48
        args.model_size = "large"
        args.ckpt_dir += f"/model.ckpt-{args.ckpt_iter}"

    # *** SELECT HERE THE LAYER INDICES TO EXTRACT: WE EXTRACT ALL THE LAYERS BY DEFAULT
    # *** THIS CAN REQUIRE A LOT OF CPU-MEMORY SINCE ALL THE DATA REPRESENTATIONS
    # *** (n_layers x nsamples x embedding) are in RAM
    args.layers_to_extract = np.arange(args.n_layer + 1)
    args.embdim = {layer: args.n_embd for layer in range(args.n_layer + 1)}
    print(f"layers to extract:  {args.layers_to_extract}")

    hparams = set_hparams(args)
    (
        trainable_params,
        gen_logits,
        gen_loss,
        clf_loss,
        tot_loss,
        accuracy,
        activations,
    ) = create_model(x, y, args.n_gpu, hparams)

    reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, args.n_gpu)

    saver = tf.train.Saver(
        var_list=[tp for tp in trainable_params if not "clf" in tp.name]
    )
    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    ) as sess:
        sess.run(tf.global_variables_initializer())

        # *** THE LOGIT OF THIS PART IS SLIGHTLY DIFFERENT THAN THE ORIGINAL
        # *** we allow the analysis of an untrained model
        # *** we analyze one folt at a time (train=default, val, test)
        # *** we added the fill data function
        desc = "_untrained"
        if not args.not_trained:
            # *** LOAD MODEL
            saver.restore(sess, args.ckpt_dir)
            print(f"restored model from ckpt {args.ckpt_dir}")
            sys.stdout.flush()
            desc = ""
            if args.ckpt_iter != 1000000:
                desc = f"_cpt{args.ckpt_iter}"
                print(f"using checkpoint {args.ckpt_iter}")

        # ***WE ONLY EXTRACT ONE DATASET AT A TIME
        Xdata, Ydata, description = load_data(args)
        description += desc

        # ***FILL DATA FUNCTION
        nsamples = Xdata.shape[0]
        Xdata, Ydata = fill_data(
            Xdata, Ydata, args
        )  # workaround to set number of data equal divisible by nbatch

        assert (
            Xdata.shape[0] % n_batch == 0
        )  # make sure that the nuber of batches fills the array

        # if args.eval:
        evaluate(
            sess,
            Xdata,
            Ydata,
            X,
            Y,
            gen_loss,
            clf_loss,
            accuracy,
            n_batch,
            activations,
            nsamples,
            args,
            description,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
