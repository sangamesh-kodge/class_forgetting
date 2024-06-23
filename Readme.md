# Introduction
This is official repository for the paper [Deep Unlearning: Fast and Efficient Gradient-free Approach to Class Forgetting](https://arxiv.org/abs/2312.00761).

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
Kodge, S., Saha, G., & Roy, K. (2023). Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting. arXiv preprint arXiv:2312.00761.
```

### Bibtex
```
@article{kodge2023deep,
  title={Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting},
  author={Kodge, Sangamesh and Saha, Gobinda and Roy, Kaushik},
  journal={arXiv preprint arXiv:2312.00761},
  year={2023}
}
```
