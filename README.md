# Neural Networks From Scratch to Framework

A learning project that intends to evolve from a simple neural network built from scratch, to a PyTorch based framework.

1. [Results](#results)
2. [Running code](#running-code)
3. [System Setup](#system-setup)



## Results
| Dataset  | Fully Connected |
|----------|-----------------|
| MNIST    | **98.12%**      |

- Manual backpropagation, forward pass, stochastic gradient descent, and parameters update.
- Only dependent on NumPy.
- Efficient memory usage: It re-utilizes pre-allocated buffers to avoid alloc/free requests during training.
- Pluggable interface to define different activation, decision and loss functions.
 

### Test accuracy during training.
``` shell
$ python main.py
TRAINING NETWORK FROM LAYER'S DESCRIPTION
in [784, 200, 200, 10] out
Epoch   1:  77.48%
Epoch   2:  89.66%
Epoch   3:  91.38%
Epoch   4:  92.48%
Epoch   5:  93.53%
Epoch   6:  94.25%
Epoch   7:  94.72%
Epoch   8:  95.17%
...
Epoch  92:  98.01%
Epoch  93:  98.03%
Epoch  94:  98.02%
Epoch  95:  98.02%
Epoch  96:  98.03%
Epoch  97:  98.03%
Epoch  98:  98.12% <--- best
Epoch  99:  98.06%
Epoch 100:  98.01%
DONE!
```



## Running code
Place the `mnist.plk.gz` file at the root directory of this repo and then run the main file.

``` shell
$ ls mnist.pkl.gz
mnist.pkl.gz
$ python main.py
```



## Setup
We just need python and numpy, but for reproducible results, let's pack things in a conda environment.

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
# go to this repo's directory and check the conda environment.yml
$ cd path/to/this/repo

# take a look at the provided conda environment specification. 
$ cat conda-environment.yml

# then, actually create the environment.
$ conda env create -f conda-environment.yml

# activate it.
$ conda activate numpy-env
```
