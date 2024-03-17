# The geometry of hidden representations of large transformer models

Source code of the paper  The geometry of hidden representations of large transformer models


## Platforms:
- Ubuntu 22.04



## Premise

The results of the paper rely on intrinsic dimension and neighborhood overlap computation.
We use the implementation of intrinsic dimension and neighborhood overlap of [Dadapy](https://github.com/sissa-data-science/DADApy).

With Dadapy, you can compute the intrinsic dimension (Fig. 1, 3) of the dataset representation at a given layer X as follows (X=2d-numpy array with shape: N x d and N = number of examples, d = number of features):

```python
from dadapy.data import Data

# initialize the "Data" class with the set of coordinates
data = Data(X)

# compute the intrinsic dimension using 2nn estimator
id_list_2NN, _, _ = data.return_id_scaling_2NN()

# compute the intrinsic dimension up to the 64th nearest neighbors using Gride
id_list_gride, _, _ = data.return_id_scaling_gride()
```
The two methods provide similar results; you can choose either of them. The second is slightly faster and more robust. 
Check Appendix B of the [paper](https://arxiv.org/pdf/2302.00294.pdf) for the implementation choices we made to compute the ID.


The overlap with the labels Y (Fig. 4) can be computed as:

```python
overlap_labels = data.return_label_overlap(Y)

```
In this case, Y (shape: N) is a 1d-numpy array containing the integer class label associated with each example.

In the paper, we also compute the overlap between pairs of representations (Fig. 2). If X2 is a second representation (shape: N x d2), the overlap between X and X2 can be computed as:

```python
overlap_X2 = data.return_data_overlap(X2)

```
<br>
<br>


In the following, we provide the code to reproduce the paper's results.

In **1. Reproduce the paper plots** the code reproduces some paper plots starting from some precomputed statistics (mu_ratios for ID and nearest neighbor indices for the overlap). We use some Dadapy functions to compute ID and overlap for iGPT.<br>
For esm2, we directly provide the precomputed IDs and overlap with labels.

In **2. Extract the representations** the code extracts the distance matrices required for the ID and overlap computation from iGPT.

## 1. Reproduce the paper plots

### a. Build an environment with the required dependencies

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html. Then, install the dependencies:

```
conda create -n geometry_representations python=3.11 pip
conda activate geometry_representations
pip install -r requirements.txt   
```

### b. Download the computed intrinsic dimension and overlaps. 
The download.py script downloads the numpy arrays needed to reproduce the plots shown in the paper.

```
python download.py 
```

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




## 2. Extract the representations and compute the distance matrices of iGPT

#### a. Download the i-gpt pretrained models following the instructions at https://github.com/openai/image-gpt
   - the conda environment and dependencies differ from those used above to reproduce the paper plot. Please stick to the package versions required by **https://github.com/openai/image-gpt**
     
#### b. Compute the nearest neighbor matrix of a hidden layer representation:

    ```
    python  get_distance_matrices.py --model small --ckpt_path "model_folder" --data_path "imagenet_folder" --trainset --hidden_repr 16
    ```
*--ckpt_path* is the directory where you stored the model checkpoints downloaded in 1.;

*--data_path* is the directory where you stored the ImageNet dataset as downloaded in 1. In the paper, we analyzed 90000 inputs from the ImageNet **training set**; 

*--hidden_repr* is the layer you want to extract.








