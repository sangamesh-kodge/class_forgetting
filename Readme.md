# Introduction
This is official repository for the paper [Deep Unlearning: Fast and Efficient Gradient-free Approach to Class Forgetting](https://openreview.net/forum?id=BmI5p6wBi0) accepted at TMLR.

# Setup Environment using yml
```bash
conda env create -f env.yml
conda activate forget
```
# Demo of unlearning algorithm 
```bash
python3 ./demo.py
```
#  Unlearning Single class on CIFAR10, CIFAR100 and ImageNet.
```bash
# for CIFAR10
sh ./scripts/our_cifar10.sh
```

```bash
# for CIFAR100
sh ./scripts/our_cifar100.sh
```

```bash
# for ImageNet
sh ./scripts/our_imagenet.sh
```
#  Analysis
Scripts for analysis done in the paper can be found in ```scripts/analysis```. 

An older version of the repository can be found in the legacy branch of the repository. 

# Citation
Kindly cite the paper if you use the code. Thanks!

### APA
```
Kodge, Sangamesh, Gobinda Saha, and Kaushik Roy. "Deep Unlearning: Fast and Efficient Gradient-Free Class Forgetting."
```

### Bibtex
```
@article{
kodge2024deep,
title={Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting},
author={Sangamesh Kodge and Gobinda Saha and Kaushik Roy},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=BmI5p6wBi0},
note={}
}
```
