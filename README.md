# The geometry of hidden representations of large transformer models

Source code of the paper:  '[The geometry of hidden representations of large transformer models](https://arxiv.org/abs/2302.00294)'. 
This work has been included in the [NeurIPS 2023 proceedings](https://papers.nips.cc/paper_files/paper/2023/hash/a0e66093d7168b40246af1cddc025daa-Abstract-Conference.html).


## Platform
- Ubuntu 22.04


## Premise

The results of the paper rely on intrinsic dimension and neighborhood overlap computation.
We use the implementation of intrinsic dimension and neighborhood overlap of [DADApy](https://github.com/sissa-data-science/DADApy).

With Dadapy, you can compute the intrinsic dimension (Fig. 1, 3) of the dataset representation at a given layer X as follows:

```python
from dadapy.data import Data

# Initialize the Data class with the layer representation X.
# X must be 2d-numpy array with shape N x d. N is the dataset size, and d is the embedding dimension. 
data = Data(X)

# compute the intrinsic dimension using 2nn estimator
id_list_2NN, _, _ = data.return_id_scaling_2NN()

# compute the intrinsic dimension up to the 64th nearest neighbors using Gride
id_list_gride, _, _ = data.return_id_scaling_gride()
```
The two methods provide similar results; you can choose either of them. The second is slightly faster and more robust. 
The above methods output a list of intrinsic dimensions; check Appendix B of the [paper](https://arxiv.org/pdf/2302.00294.pdf) to see how we select the ID we plot in the figures.


The overlap with the labels Y (Fig. 4) can be computed as:

```python
# Y (shape: N) must be a 1d-numpy array with the integer class label of each example.
overlap_labels = data.return_label_overlap(Y, k=30)
```

In the paper, we also compute the overlap between pairs of representations (Fig. 2). If X2 is a second representation (shape: N x d2), the overlap between X and X2 can be computed as:

```python
overlap_X2 = data.return_data_overlap(X2, k=30)
```
<br>


In the following, we provide the code to reproduce the paper's results.

In **1. Reproduce the paper plots**, the code reproduces some paper plots starting from some precomputed statistics (mu_ratios for ID and nearest neighbor indices for the overlaps. See the Method section of the [paper](https://arxiv.org/pdf/2302.00294.pdf) for the meaning of these quantities). We use some Dadapy functions to compute ID and overlap for iGPT.<br>
For esm2, we directly provide the precomputed IDs and overlap with labels.

In **2. Extract the representations**, the code extracts the distance matrices required for the ID and overlap computation from iGPT.



<br>
<br>

## 1. Reproduce the paper plots

### a. Build an environment with the required dependencies

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html. Then, install the dependencies:

```
conda create -n geometry_representations python=3.11 pip
conda activate geometry_representations
pip install -r requirements.txt   
```
<br>

### b. Download the computed intrinsic dimension and overlaps. 
The download.py script downloads the numpy arrays needed to reproduce the plots shown in the paper.

```
python download.py 
```

<br>

### c. Plot the intrinsic dimension and overlap profiles
You can plot the intrinsic dimension profiles (Fig. 1) and the overlap with the class labels (Fig. 4).

```
python plot_id_overlap.py 
```
The plots shown below are saved by default in "./results/plots".

<table>
  <tr>
    <td><img src=results/plots/igpt_id_ov_labels.png width="250"></td>
    <td><img src=results/plots/esm_id_ov_labels.png width="250"></td>
  </tr>
</table>


The iGPT intrinsic dimension and overlaps are computed using the nearest neighbors' distance matrices you downloaded in 2. For the neighborhood overlap (bottom-right), we provide only a small number of checkpoints.

You can use the code from the following section to extract the distance matrices of all the layers in iGPT.


<br>
<br>

## 2. Extract the representations and compute the distance matrices of iGPT

We first need to download the pre-trained iGPT models following the instructions in https://github.com/openai/image-gpt. The environment with the required dependencies can be created as follows:

```
conda create --name image-gpt python=3.7.3 pip
conda activate image-gpt

conda install numpy=1.16.3
conda install tensorflow-gpu=1.13.1

conda install imageio=2.8.0
conda install requests=2.21.0
conda install tqdm=4.46.0
pip install -U scikit-learn
```
We added the scikit-learn package to the environment of https://github.com/openai/image-gpt. We use it to compute the distance matrices.

<br>

#### a. Download the i-gpt pretrained models.
You can download the iGPT-small model and the ImageNet dataset (training, validation, test sets) with:
```
python src/download_igpt.py
--model s
--ckpt 1000000
--dataset imagenet
--download_dir igpt_models
```

*--model 's'*  means that we download the small version of iGPT; <br>
*--ckpt 1000000* is the training checkpoint to download. 1 000 000 means fully trained model; <br>
*--dataset imagenet* means that we download the ImageNet dataset; <br>
*--download_dir igpt_models*  is the directory where the pre-trained model and ImageNet dataset are stored. <br>

The disk memory occupied by models and datasets is as follows: <br>
iGPT-small: 894MB; <br>
iGPT-medium: 5.2GB; <br>
iGPT-large 15.5GB; <br> 
ImageNet dataset 11GB. 

<br>

#### b. Compute the nearest neighbor matrix. 
With the following command, you will extract the 24 hidden representations of iGPT-small analyzed in the paper. 

In a V100 GPU with 32GB of V-RAM, extracting the representations of 90k examples takes around one hour and requires 32GB of V-RAM (with a batch size of 8). You can decrease the GPU memory requirement by decreasing the batch size. After the representations have been extracted, the computation of the 24 distance matrices takes another 30 minutes (for 90k examples). 

The representations are stored in RAM during the process. Depending on your memory budget, you may need to extract smaller subsets of layers. 

```
python  src/run.py 
--data_dir igpt_models 
--ckpt_dir igpt_models 
--model "s" 
--results_dir "./results" 
--nimg_cat 300 
--n_sub_batch 8 
```

*--ckpt_path* is the directory where you stored the model checkpoints downloaded in a.; <br>
*--model* 's' means that you are analyzing the small model; <br>
*--data_dir* is the directory where you stored the ImageNet dataset as downloaded in a.; <br>
*--results_dir* is the directory where the representations/distance matrices are saved; <br>
*--nimg_cat* is the number of images per class analyzed (300 in the paper); <br>
*--n_sub_batch*  is the batch size. <br>

In the run.py, we extract only the 300 classes from the ImageNet **TRAINING SET** analyzed in the paper. <br>
The class labels are stored in the './hier_nucl_labels.npy' array. 

<br>

#### c. Extract the hidden layer representations.

If you just want to extract the hidden layer representations, add the *--activations* argument to the previous ones:

```
python  src/run.py 
--activations 
--data_dir igpt_models 
--ckpt_dir igpt_models 
--model "s" 
--results_dir "./results" 
--nimg_cat 300 
--n_sub_batch 8 
```

With this setup, the distance matrices are not computed. 
