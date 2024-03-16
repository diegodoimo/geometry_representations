# The geometry of hidden representations of large transformer models

Source code of the paper  The geometry of hidden representations of large transformer models


## Platforms:
- Ubuntu 22.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html. Then, install the dependencies shown below manually.

```
conda create -n  geometry_representations python=3.11 pip
conda activate geometry_representations
pip install -r requirements.txt   
```

## Reproduce the plots of the paper
### Download the computed intrinsic dimension and overlaps. 
The download.py script downloads the numpy arrays needed to reproduce the plots shown in the paper.

```
python download.py 
```

## Plot the intrinsic dimension and overlap profiles
### Image
With the following, you can plot the intrinsic dimension profiles (Fig. 1) and the overlap with the class labels (Fig. 4). The plots shown below are saved by default in "./results/plots"

```
python plot_id_overlap.py 
```

<table>
  <tr>
    <td><img src=results/plots/igpt_id_ov_labels.png width="200"></td>
    <td><img src=results/plots/esm_id_ov_labels.png width="200"></td>
  </tr>
</table>



## Compute the distance matrices 
### Image

If you'd like to compute yourself the distance matrix of other hidden representations of the ImageNet dataset:

   * #### Download the i-gpt pretrained models following the instructions at https://github.com/openai/image-gpt 


   * #### Compute the nearest neighbor matrix of an hidden layer representation:

   To compute the nearest neighbor matrix, you need to store the pre-trained model weights in "model_folder" and the ImageNet dataset in "imagenet_folder." 

   To compute the distance matrix of the 'layer 16' of the iGPT 'small' architecture on the ImageNet training set type:

    ```
    python  get_distance_matrices.py --model small --ckpt_path "model_folder" --data_path "model_folder" --trainset --hidden_repr 16
    ```

### Protein 








