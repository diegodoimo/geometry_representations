# The geometry of hidden representations of large transformer models

Source code of the paper  The geometry of hidden representations of large transformer models


## Platforms:
- Ubuntu 22.04


## Reproduce the plots of the paper

### 1. Build an environment with the required dependencies

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html. Then, install the dependencies shown below manually.

```
conda create -n  geometry_representations python=3.11 pip
conda activate geometry_representations
pip install -r requirements.txt   
```

### 2. Download the computed intrinsic dimension and overlaps. 
The download.py script downloads the numpy arrays needed to reproduce the plots shown in the paper.

```
python download.py 
```

### 3. Plot the intrinsic dimension and overlap profiles
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




## Extract the representations and compute the distance matrices of iGPT

1. #### Download the i-gpt pretrained models following the instructions at https://github.com/openai/image-gpt
   - the conda environment and dependencies differ from those used above to reproduce the paper plot. Please stick to the package versions required by **https://github.com/openai/image-gpt**
     
2. #### Compute the nearest neighbor matrix of a hidden layer representation:
    ```
    python  get_distance_matrices.py --model small --ckpt_path "model_folder" --data_path "imagenet_folder" --trainset --hidden_repr 16
    ```
*--ckpt_path* is the directory where you stored the model checkpoints downloaded in 1.;

*--data_path* is the directory where you stored the ImageNet dataset as downloaded in 1. In the paper, we analyzed 90000 inputs from the ImageNet **training set**; 

*--hidden_repr* is the layer you want to extract.








