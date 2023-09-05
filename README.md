# LG-GNN
This repo is the official implementation of [Classification of Brain Disorders in rs-fMRI via Local-to-Global Graph Neural Networks](https://ieeexplore.ieee.org/document/9936686)
<div align=center/> <img width="688" alt="image" src="https://user-images.githubusercontent.com/43660513/205473700-377f514a-c6dc-42ed-92a8-9c766399238f.png">

## I. Usage:
The main contribution of our work is the proposed LG-GNN architecture, which enables crosstalk between local brain regions and the global population, and allows identification biomarkers.

The data used in our work are from [ADNI](https://adni.loni.usc.edu/) and [ABIDE](http://preprocessed-connectomes-project.org/abide/). Please follow the relevant regulations to download from the websites.

Using the `main.py` to train and test the model on your own dataset.
The proposed network **LG-GNN** is defined in the `layer.py`.  It can be easily edited and embed in your own code.

## II. Requirements:
* torch_geometric
* torch
* scipy
* visdom
* numpy
* os
## III. Citation：
If our paper or code is helpful to you, please cite our paper. If you have any questions, please feel free to ask me.
```
@article{zhang2023classification,
  title={Classification of brain disorders in rs-fMRI via local-to-global graph neural networks},
  author={Zhang, Hao and Song, Ran and Wang, Liping and Zhang, Lin and Wang, Dawei and Wang, Cong and Zhang, Wei},
  journal={IEEE Transactions on Medical Imaging},
  volume={42},
  number={2},
  pages={444--455},
  year={2023},
  publisher={IEEE}
}
```

