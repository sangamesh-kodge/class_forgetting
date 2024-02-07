# Setup Environment using yml
```bash
conda env create -f env.yml
conda activate forget
```
# Demo of unlearning algorithm 
```bash
python3 ./class_removal/demo.py
```
#  Unlearning Single class on CIFAR10, CIFAR100 and ImageNet.
```bash
# for CIFAR10
sh ./class_removal/run_cifar10.sh
```

```bash
# for CIFAR100
sh ./class_removal/run_cifar100.sh
```

```bash
# for ImageNet
sh ./class_removal/run_imagenet.sh
```

# Citation
Kindly cite the paper if you use the code. Thanks!

### APA
```
Kodge, S., Saha, G., & Roy, K. (2023). Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting. https://doi.org/10.48550/arXiv.2312.00761
```

### Bibtex
```
@misc{kodge2023deep,
      title={Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting}, 
      author={Sangamesh Kodge and Gobinda Saha and Kaushik Roy},
      year={2023},
      eprint={2312.00761},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```