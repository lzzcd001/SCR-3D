# Structural Causal 3D Reconstruction

By Weiyang Liu*, Zhen Liu*, Liam Paull, Adrian Weller, Bernhard Schölkopf

### Introduction

This paper considers the problem of unsupervised 3D object reconstruction from in-the-wild single-view images. Due to ambiguity and intrinsic ill-posedness, this problem is inherently difficult to solve and therefore requires strong regularization to achieve disentanglement of different latent factors. Unlike existing works that introduce explicit regularizations into objective functions, we look into a different space for implicit regularization – the structure of latent space. Specifically, we restrict the structure of latent space to capture a topological causal ordering of latent factors (i.e., representing causal dependency as a directed acyclic graph). We first show that different causal orderings matter for 3D reconstruction, and then explore several approaches to find a task-dependent causal factor ordering. Our experiments demonstrate that the latent space structure indeed serves as an implicit regularization and introduces an inductive bias beneficial for reconstruction.

### Get started

Simply run the script with

```Shell
python run.py
```

### Citation
If you find our work useful in your research, please consider to cite:

    @InProceedings{Liu2022SCR,
      title={Structural Causal 3D Reconstruction},
      author={Liu, Weiyang and Liu, Zhen and Paull, Liam and Weller, Adrian and Schölkopf, Bernhard},
      booktitle = {ECCV},
      year={2022}
    }Un

This repository is built largely based on the great work of [unsup3d](https://github.com/elliottwu/unsup3d).
