## Element-Wise Activation Scaling (EWAS)
Official code for [Improving Robustness of Convolutional Neural Networks Using Element-Wise Activation Scaling]()


![EWAS](figures/my-proposed.png)

--------------------
### Requirements

- torch==1.6.0
- torchvision==0.7.0
- tqdm

---

### How to run?

First, you need to create one empty folder and name it as `data` to store the dataset.

run the command
```commandline
python AT_with_loss.py --net wrn_ras --lam 0.01
python AT_with_loss.py --net resnet_ras --lam 0.01
python MART_with_loss.py --net wrn_ras --lam 0.01
python MART_with_loss.py --net resnet_ras --lam 0.01 
python TRADES_with_loss.py --net wrn_ras --lam 0.01
python TRADES_with_loss.py --net resnet_ras --lam 0.01 
```

The results are stored in folds named with adversarial training methods and name with model name and training lambda.

### Evaluation

run the command
```commandline
python white_box_robust_eval.py --net resnet_ras --ckps_path './AT_results/last_rap-resnet18_lam0.01.pth.tar' --lam 0.01
...
```

The results of CIFAR10 as follows:

![CIFAR10](figures/cifar10.png)
