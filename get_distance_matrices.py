
import argparse
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf
import pathlib


from tensorflow.contrib.training import HParams
from src.model import model
from src.utils import iter_data, count_parameters
from src import pairwise_distance
import os, psutil
p = psutil.Process(os.getpid())


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data and I/O

    parser.add_argument("--data_path", type=str, default="./dataset/cifar10")
    #parser.add_argument("--ckpt_path", type=str, default="./models/model.ckpt-100000")
    parser.add_argument("--ckpt_iter", type=int, default=1000000)
    parser.add_argument("--ckpt_path", type=str, default="./models")
    parser.add_argument("--color_cluster_path", type=str, default="./dataset/kmeans_centers.npy")
    parser.add_argument("--save_dir", type=str, default="./results/save/")

    # model
    parser.add_argument("--model", type=str, choices=["s", "m", "l"], default = "s")
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=24)
    parser.add_argument("--n_px", type=int, default=32, help="image height or width in pixels")
    parser.add_argument("--n_vocab", type=int, default=512, help="possible values for each pixel")
    
    parser.add_argument("--bert", action="store_true", help="use the bert objective (defaut: autoregressive)")
    parser.add_argument("--bert_mask_prob", type=float, default=0.15)
    parser.add_argument("--clf", action="store_true", help="add a learnable classification head")

    # parallelism
    #originally 8 8
    parser.add_argument("--n_sub_batch", type=int, default=1, help="per-gpu batch size")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus to distribute training across")

    # mode
    parser.add_argument("--eval", action="store_true", help="evaluates the model, requires a checkpoint and dataset")
    parser.add_argument("--eval_accuracy", action="store_true", help="evaluates the model, requires a checkpoint and dataset")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="seed for random, np, tf")

    #hidden repr estraction
    parser.add_argument("--trainset", action = "store_true")
    parser.add_argument("--subset", action = "store_true")
    parser.add_argument("--valset", action = "store_true")
    parser.add_argument("--save_act", action = "store_true")
    parser.add_argument("--save_labels", action = "store_true")
    parser.add_argument("--not_trained", action = "store_true")
    parser.add_argument("--save_gen_losses", action = "store_true")
    parser.add_argument("--hidden_repr", type=int, default=0)
    parser.add_argument("--maxk", type=int, default=50)
    parser.add_argument("--working_memory", type=int, default=1024)
    parser.add_argument("--results_folder", default = './results')

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def load_data(args):
    data_path = args.data_path
    if args.trainset:
        Xdata = np.load(f'{data_path}_trX.npy')
        Ydata = np.load(f'{data_path}_trY.npy')
        description = 'train'

    elif args.valset:
        Xdata = np.load(f'{data_path}_vaX.npy')
        Ydata = np.load(f'{data_path}_vaY.npy')
        description = 'valid'

    else:
        Xdata = np.load(f'{data_path}_teX.npy')
        Ydata = np.load(f'{data_path}_teY.npy')
        description = 'test'

    if args.subset:                                      #take a subset of dataste accordine to labels (hierarchical nucleations training set)
        labels = np.load('./hier_nucl_labels.npy')
        full_targets = np.where(Ydata)[1]                #get real numbers from one hot
        subset_indices = []                              #indices new subset
        for i in range(len(labels)):
            subset_indices.extend( np.where(full_targets == labels[i])[0][:300] )

        subset_indices = np.array(subset_indices)
        print('initial training shapes', Xdata.shape, Ydata.shape)
        Xdata = Xdata[subset_indices]
        Ydata = Ydata[subset_indices]
        print('final training shapes', Xdata.shape, Ydata.shape)
        description+='_300cl_300im'

    return Xdata, Ydata, description

def fill_data(Xdata, Ydata, args):
    n_batch = args.n_sub_batch * args.n_gpu
    nfiller = (n_batch - Xdata.shape[0]%n_batch)%n_batch
    Xfiller = np.ones((nfiller, Xdata.shape[1]), dtype = Xdata.dtype)
    Yfiller = np.zeros((nfiller, Ydata.shape[1]), dtype = Ydata.dtype)
    Xdata = np.vstack((Xdata, Xfiller))
    Ydata = np.vstack((Ydata, Yfiller))
    return Xdata, Ydata

def set_hparams(args):
    return HParams(
        n_ctx=args.n_px*args.n_px,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_vocab=args.n_vocab,
        bert=args.bert,
        bert_mask_prob=args.bert_mask_prob,
        clf=args.clf,
        hidden_repr = args.hidden_repr,
    )


def create_model(x, y, n_gpu, hparams):

    gen_loss = []
    stats = []
    clf_loss = []
    tot_loss = []
    accuracy = []
    activations = []

    trainable_params = None
    for i in range(n_gpu):
        with tf.device("/gpu:%d" % i):
            print(hparams)
            results = model(hparams, x[i], y[i], reuse=(i != 0))

            gen_loss.append(results["gen_loss"])
            stats.append(results["stats"])
            clf_loss.append(results["clf_loss"])

            if hparams.clf:
                tot_loss.append(results["gen_loss"] + results["clf_loss"])
            else:
                tot_loss.append(results["gen_loss"])

            accuracy.append(results["accuracy"])
            activations.append(results["activations"])

            if i == 0:
                trainable_params = tf.trainable_variables()
                print("trainable parameters:", count_parameters())

    return trainable_params, gen_loss, stats, clf_loss, tot_loss, accuracy, activations


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


def evaluate(sess, evX, evY, X, Y, gen_loss, stats, clf_loss, accuracy, activations, n_batch, nsamples, desc, args):

    metrics = []
    labels = np.where(evY)[1]

    for i, (xmb, ymb) in enumerate(iter_data(evX, evY, n_batch=n_batch, truncate=False, verbose=True)):
        if (i+1)%20==0:
            print(f"{i*n_batch/1000}k/{nsamples}")
            sys.stdout.flush()
        metrics.append(     sess.run([gen_loss[0], stats[0], clf_loss[0], accuracy[0], activations[0]], {X: xmb, Y: ymb})   )


    gen_loss, _, _, _, stack = list(zip( *metrics  ))           #tuple of N arrays of shape 1x512

    act = np.vstack((stack))                                    #Nx1x512
    act = act[:nsamples]
    print('matrix shape', act.shape)

    gen_loss = np.mean( np.vstack(gen_loss)[:nsamples] )        #Nx1x512
    labels = labels[:nsamples]

    print('shape activations:', act.shape)
    if len(act.shape)>2:                                        #some axis equal 1
        act = np.squeeze(act)                                   #Nx512

    print('Compute distance matrix')
    act = act.astype(np.float64, casting="safe")
    #"compute distances and ms for id scaling"
    distances, dist_index, mus, rs = pairwise_distance.compute_distances(
                             X = act,
                             n_neighbors=args.maxk+1,
                             n_jobs=1,
                             working_memory = args.working_memory,
                             range_scaling = 1050,
                             argsort=False
                             )
    
    print('Saving distance matrix')
    if not os.path.isdir(f'{args.results_folder}/{args.model_size}'):
        pathlib.Path(f'{args.results_folder}/{args.model_size}').mkdir(parents=True, exist_ok=True)

    if args.data_path.endswith("imagenet"):
        dataset = 'imagenet'
    elif args.data_path.endswith("cifar10"):
        dataset = 'cifar10'
    
    np.save(f'{args.results_folder}/{args.model_size}/{dataset}_{args.model_size}_layer{args.hidden_repr}_{desc}_dist' , distances)
    np.save(f'{args.results_folder}/{args.model_size}/{dataset}_{args.model_size}_layer{args.hidden_repr}_{desc}_index' , dist_index)
    np.save(f'{args.results_folder}/{args.model_size}/{dataset}_labels.npy', labels)

    eval_gen_loss, _, eval_clf_loss, eval_accuracy, _ = [np.mean(m) for m in zip( *metrics  )]
    print(f"{desc} gen: {eval_gen_loss:.4f} clf: {eval_clf_loss:.4f} acc: {eval_accuracy:.2f}")



def main(args):

    set_seed(args.seed)

    n_batch = args.n_sub_batch * args.n_gpu

    if args.data_path.endswith("cifar10"):
        print("dataset = cifr10")
        n_class = 10
    elif args.data_path.endswith("imagenet"):
        print("dataset = imagenet")
        n_class = 1000
    else:
        raise ValueError("Dataset not supported.")

    X = tf.placeholder(tf.int32, [n_batch, args.n_px * args.n_px])
    Y = tf.placeholder(tf.float32, [n_batch, n_class])

    x = tf.split(X, args.n_gpu, 0)
    y = tf.split(Y, args.n_gpu, 0)

    if args.model == 's':
        args.n_embd = 512
        args.n_head = 8
        args.n_layer = 24
        args.model_size = 'small'
        args.ckpt_path += f"/small/model.ckpt-{args.ckpt_iter}"
    elif args.model == 'm':
        args.n_embd = 1024
        args.n_head = 8
        args.n_layer = 36
        args.model_size = 'medium'
        args.ckpt_path += f"/medium/model.ckpt-{args.ckpt_iter}"
    elif args.model == 'l':
        args.n_embd = 1536
        args.n_head = 16
        args.n_layer = 48
        args.model_size = 'large'
        args.ckpt_path += f"/big/model.ckpt-{args.ckpt_iter}"

    hparams = set_hparams(args)
    trainable_params, gen_loss, stats, clf_loss, tot_loss, accuracy, activations = create_model(x, y, args.n_gpu, hparams)

    reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, args.n_gpu)

    saver = tf.train.Saver(var_list=[tp for tp in trainable_params if not 'clf' in tp.name])
    print(' model dataset init finished')

    sys.stdout.flush()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        print('ok')
        sys.stdout.flush()

        sess.run( tf.global_variables_initializer() )

        desc='_untrained'
        if not args.not_trained:
            saver.restore(sess, args.ckpt_path)
            print(f'restored model from ckpt {args.ckpt_path}')
            sys.stdout.flush() 
            desc = ''
            if args.ckpt_iter!=1000000:
                desc = f'_cpt{args.ckpt_iter}'
                print(f'using checkpoint {args.ckpt_iter}')

        sys.stdout.flush()
        Xdata, Ydata, description = load_data(args)

        description+=desc

        print('original data shape:', Xdata.shape, n_batch, Xdata.shape[0]%n_batch)
        sys.stdout.flush()
        nsamples = Xdata.shape[0]
        Xdata, Ydata = fill_data(Xdata, Ydata, args)    #workaround to set number of data equal divisible by nbatch

        assert Xdata.shape[0]%n_batch == 0              #make sure that the nuber of batches fills the array

        #compute_distance_matrices
        evaluate(sess, Xdata, Ydata, X, Y, gen_loss, stats, clf_loss, accuracy, activations, n_batch,  nsamples, description, args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
