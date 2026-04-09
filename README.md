# Neural Networks, From Scratch to Framework

A learning project that intends to evolve from a simple neural network built from scratch, to a PyTorch based framework.

1. [Results](#results)
2. [Running the code](#running-the-code)
3. [Changelog Summary](#changelog-summary)
4. [System Setup](#system-setup)



## Results
| Dataset  | Fully Connected |
|----------|-----------------|
| MNIST    | **98.23%**      |

### Test accuracy during training.
``` shell
$ python main.py                                                                                      (torch-xpu-0) 
RUNNING: main.py

PROBING SYSTEM
XPU available?: True

LOADING DATA
Training:(X,Y):
    X    torch.float32   [50000, 784]
    Y      torch.int32   [50000, 1]
Validation:(X,Y):
    X    torch.float32   [10000, 784]
    Y      torch.int32   [10000, 1]
Test:(X,Y):
    X    torch.float32   [10000, 784]
    Y      torch.int32   [10000, 1]

CREATING NETWORK
in -> [784, 200, 200, 10] -> out

TRAINING
  epochs     : 100
  batch sz   : 20
  learn rate : 0.5
Epoch   1:  93.30%
Epoch   2:  95.70%
Epoch   3:  96.57%
Epoch   4:  96.86%
Epoch   5:  97.21%
Epoch   6:  97.18%
Epoch   7:  97.14%
Epoch   8:  97.47%
...
Epoch  17:  98.23% <--- best
...
Epoch  92:  98.13%
Epoch  93:  98.10%
Epoch  94:  98.10%
Epoch  95:  98.13%
Epoch  96:  98.14%
Epoch  97:  98.12%
Epoch  98:  98.09%
Epoch  99:  98.09%
Epoch 100:  98.13%
SAVING W+B TO main_experiment.pth
DONE!
```

## Running the code
Place the `mnist.plk.gz` file at the root directory of this repo (you can get mnist [from here](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/refs/heads/master/data/mnist.pkl.gz)), activate the conda environment (see [System Setup](#system-setup) section), and then run the main file.

``` shell
$ ls mnist.pkl.gz
mnist.pkl.gz
$ conda activate torch-xpu
$ python main.py
```

## Changelog Summary
- Version 6.0.0:
  - Replaces Mean Square Error with Cross Entropy, so model shows steeper gradients when "confidently wrong". This is evidenced by the model quickly learning at the beginning of the training, reaching its plateau at as early as epoch 17.

- [Version 5.0.0](../../tree/v5.0.0):
  - Replace manual tensor handling of model's layers with `torch.nn.ModuleList` and `nn.Linear`.
  - Replace manual (and freely named) `NeuralNetwork._feedforward` method with specific `.forward()` method that integrates with torch system.
  - Replace manual xavier and zero initializations with in-place `torch.init.xavier_` and `.zero_`, respectively.
  - Replace manual stochastic gradient descent with `torch.optim.SGD`. As a consequence, updating the model's parameters simplifies to `sgd.step()`

- [Version 4.0.0](../../tree/v4.0.0):
  - Replace manual backpropagation with Torch Autograd.
  - Now training memory is managed by Autograd.

- [Version 3.0.0](../../tree/v3.0.0):
  - Parallel batched training. As a result, the model trains significantly faster than the previous version.
  - Separates memory required for evaluation and training modes, introducing a `Workspace` class that allocates training space only during training.

- [Version 2.0.0](../../tree/v2.0.0):
  - Migrate tensors from NumPy to PyTorch.
  - Use XPU (Intel) for the first time!
  - Profile CPU and XPU activity.

- [Version 1.0.0](../../tree/v1.0.0):
  - Manual backpropagation, forward pass, stochastic gradient descent, and parameters update.
  - Only depends on NumPy.
  - Efficient memory usage: It re-utilizes pre-allocated buffers to avoid alloc/free requests during training.
  - Pluggable interface to define different activation, decision and loss functions.


## System Setup
I am using an Intel Arc Pro B50 (on Fedora). If using CPU, or other graphic card, then you may want to setup things by yourself.

**Note:** You may need to to reboot your computer and dig into the UEFI settings. Make sure your UEFI setting "Large BAR" is enabled. Intel says the option might be labeled "Re-Size BAR" or "Smart Access Memory" depending on your motherboard maker.

**Note 2:** As of February 2026, with kernel 6.18 [there was a bug](https://github.com/pytorch/pytorch/issues/172934) that broke things when moving data from the device to the cpu . You can avoid the issue by using kernel 6.17.

### XPU Drivers and system packages
``` shell
$ sudo dnf install clinfo intel-gpu-tools intel-compute-runtime oneapi-level-zero intel-gpu-tools

# check that you actually have the card 
$ ls -l /dev/dri/
total 0
drwxr-xr-x. 2 root root         80 Jan 23 16:22 by-path/
crw-rw----+ 1 root video  226,   0 Mar 26 09:14 card0
crw-rw-rw-. 1 root render 226, 128 Mar 23 12:00 renderD128

# add yourself to the render and video groups
$ sudo usermod -aG render,video $USER

# re-login or just reboot
$ sudo reboot
```


### Install Conda
``` shell
# Download conda
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install and follow prompts. Allow init
$ bash Miniconda3-latest-Linux-x86_64.sh

# if using bash, re-source the runtime config. (use equivalent for other shells)
source ~/.bashrc

# verify conda
conda --version

```

### Conda environment, and setup testing
```shell
# go to this repo directory
$ cd path/to/this/repo

# take a look at the provided conda environment specification 
$ cat conda-environment.yml

# then, actually create the environment
$ conda env create -f conda-environment.yml

# activate it
$ conda activate torch-xpu
```
