# DNAS
Official Pytorch Implementation

# Dataset
Please prepare ImageNet16-120, CIFAR-10, CIFAR-100 and NAS-Bench-201

## search on NAS-Bench-201
```
example: using ImageNet16-120，warm-up epochs 5
CUDA_VISIBLE_DEVICES=1 python DNAS.py --cfg cfgs/ImageNet_1.yaml --warmup_epochs 5
```


## Acknowledgment
This code is highly relied on [NATS-Bench](https://github.com/D-X-Y/AutoDL-Projects). 


