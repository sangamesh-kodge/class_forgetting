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

