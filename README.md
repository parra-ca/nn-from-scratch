# Neural Networks, From Scratch to Framework

A learning project that intends to evolve from a simple neural network built from scratch, to a PyTorch based framework.

1. [Results](#results)
2. [Running the code](#running-the-code)
3. [Changelog Summary](#changelog-summary)
4. [System Setup](#system-setup)



## Results
| Dataset  | Fully Connected |
|----------|-----------------|
| MNIST    | **98.12%**      |

### Test accuracy during training.
``` shell
$ python main.py 
PROBING SYSTEM
XPU available?: True

LOADING DATA

CREATING NETWORK
in -> [784, 200, 200, 10] -> out

TRAINING
  epochs    : 100
  batch sz  : 20
  learn rate: 1
Epoch   1:  88.66%
Epoch   2:  92.31%
Epoch   3:  93.75%
Epoch   4:  94.27%
Epoch   5:  95.21%
Epoch   6:  95.64%
Epoch   7:  96.07%
Epoch   8:  96.58%
...
Epoch  49:  98.12% <--- best
...
Epoch  92:  98.03%
Epoch  93:  98.04%
Epoch  94:  98.05%
Epoch  95:  98.02%
Epoch  96:  98.01%
Epoch  97:  98.01%
Epoch  98:  98.04%
Epoch  99:  98.02%
Epoch 100:  98.01%
DONE!
```

### About the speed...
The [previous version](../../tree/v2.0.0) was making terrible use of the XPU because the operations sent to it (kernels) were too many and too small. The CPU would spend most of the time just adding the kernels to the queue (39.03%); and the time it would spend "preparing" the multiplications `aten::mm` and in-place additions `aten::add_` was about **86x longer than the operations themselves** in the XPU (1300ms on CPU vs 15ms in XPU).

Profiling 1000 samples with batch size 20, on the XPU showed this:

| Name                  | Self CPU% | Self CPU  | Self XPU | Self XPU% | # of Calls |
|-----------------------|-----------|-----------|----------|-----------|------------|
| urEnqueueKernelLaunch | 39.03%    | 1.563s    | 0.000us  | 0.00%     | 43600      |
| aten::mm              | 22.52%    | 901.789ms | 5.928ms  | 17.93%    | 8000       |
| aten::add_            | 9.94%     | 398.136ms | 9.120ms  | 27.59%    | 12300      |

Now, with parallel batches, each batch sends one kernel rather than 20. The same profile shows:

| Name                  | Self CPU% | Self CPU  | Self XPU  | Self XPU% | # of Calls |
|-----------------------|-----------|-----------|-----------|-----------|------------|
| aten::mm              | 31.74%    | 277.155ms | 666.400us | 15.61%    | 400        |
| aten::sum             | 14.51%    | 126.687ms | 249.900us | 5.85%     | 150        |
| urEnqueueKernelLaunch | 11.51%    | 100.477ms | 0.000us   | 0.00%     | 2563       |
| aten::add_            | 11.10%    | 96.885ms  | 999.600us | 23.41%    | 600        |

- Now the number of queued kernels got 17x smaller; that makes sense with batches of 20 elements, but adding some new kernels that handle operate on the batches.
- Both tables show about 70% of the CPU time. In the first case that 70% takes 2863ms. In the batched implementation, 601ms. So the CPU is spending less time to accomplish the same work (training 1000 samples)!
- Similarly, both tables show about 45% of the XPU time. In the first case it takes about 15ms while the batched version takes 2ms!

All of this shows that batching does solve (to some degree) the bottleneck at the kernel dispatch overhead (40% -> 11%), making a more efficient use of the XPU. 

However the time proportion of CPU/XPU (601ms/2ms) tells that the current configuration (remember that we are using batches of 20 MNIST images) is not even scratching the computation power of the XPU. It basically sits there waiting for the CPU to send some operation only to complete it in a blink.

## Running the code
Place the `mnist.plk.gz` file at the root directory of this repo (you can get mnist [from here](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/refs/heads/master/data/mnist.pkl.gz)), activate the conda environment (see [System Setup](#system-setup) section), and then run the main file.

``` shell
$ ls mnist.pkl.gz
mnist.pkl.gz
$ conda activate torch-xpu
$ python main.py
```


## Changelog Summary
- Version 3.0.0:
  - Parallel batched training. As a result, the model trains significantly faster than the previous version.
  - Separates memory required for evaluation and training modes, introducing a `Workspace` that are allocated only during training.

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
