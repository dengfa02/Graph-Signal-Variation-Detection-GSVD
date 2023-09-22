# Graph Signal Variation Detection（GSVD）


This repository contains the algorithm done in the work [Graph Signal Variation Detection: A novel approach for identifying and
reconstructing ship AIS tangled trajectories](https://doi.org/10.1016/j.oceaneng.2023.115452) by Chuiyi Deng et al.

The core steps of GSVD algorithm for the identification and reconstruction of tangled trajectories are listed in the file.

If you find this repository helpful, please cite our work:

```
@article{deng2023graph,
  title={Graph Signal Variation Detection: A novel approach for identifying and reconstructing ship AIS tangled trajectories},
  author={Deng, Chuiyi and Wang, Shuangxin and Liu, Jingyi and Li, Hongrui and Chu, Boce and others},
  journal={Ocean Engineering},
  volume={286},
  pages={115452},
  year={2023},
  publisher={Elsevier}
}
```

## Domains and Datasets

**Update**: The code should be directly runnable with Python 3.x. The older versions of Python are no longer supported.
Scipy error may be displayed during runtime, just update it to the latest version (e.g. 1.11.2).

The dataset folder of this repository provides two original trajectories 245539000_ori and 410050325_ori as examples, which are in different seas.

## Usage

To run GSVD algorithm on the task, one only need to run `GSVD_test.py`. You can also set the hyperparameters you want in the main function of this .py file. 
