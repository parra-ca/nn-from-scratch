# Neural Networks From Scratch to Framework

A learning project that intends to evolve from a simple neural network built from scratch, to a PyTorch based framework.

1. [Results](#results)
2. [Running the code](#running-the-code)
2. [Changelog Summary](#changelog-summary)
3. [System Setup](#system-setup)



## Results
| Dataset  | Fully Connected |
|----------|-----------------|
| MNIST    | **98.11%**      |

### Test accuracy during training.
``` shell
$ python main.py
PROBING SYSTEM
XPU available?: True

LOADING DATA
Training   (50000 samples):
    X:(torch.float32)torch.Size([784, 1]) --> y:(torch.int16)torch.Size([])
Validation (10000 samples):
    X:(torch.float32)torch.Size([784, 1]) --> y:(torch.int16)torch.Size([])
Test       (10000 samples):
    X:(torch.float32)torch.Size([784, 1]) --> y:(torch.int16)torch.Size([])

CREATING NETWORK
in -> [784, 200, 200, 10] -> out

TRAINING
Epoch   1:  78.73%
Epoch   2:  89.75%
Epoch   3:  91.67%
Epoch   4:  92.82%
Epoch   5:  93.80%
Epoch   6:  94.07%
Epoch   7:  94.65%
Epoch   8:  95.06%
...
Epoch  92:  98.07%
Epoch  93:  98.10%
Epoch  94:  98.11% <--- best
Epoch  95:  98.07%
Epoch  96:  98.07%
Epoch  97:  98.08%
Epoch  98:  98.05%
Epoch  99:  98.08%
Epoch 100:  98.04%
DONE!
```

### About the speed...
The fact that the code uses the XPU (Intel's accelerator) does not mean that the code will instantly run faster. Indeed, **it is painfully slow!** 
The reason is:
1. tensors are tiny (this is just MNIST, each image is 3KiB) and the overhead of the surrounding data structures and calling the algebra engine is too large when compared to the actual operation.
2. tensor operations are executed **one at the time**. The average of a batch gradient is computed by first accumulating (`+=`) the gradients of all the samples in a batch, in an intermediate buffer `NeuralNetwork.weights_acc_grad` and `NeuralNetwork.biases_acc_grad`, and after the batch has been completed, divide that by the number of samples in the batch, and finally update the model's parameters.
   ```python
   def stoc_gradient_descent():
       ...
       # for each epoch...
       for epoch in range(epochs):
           ...
           # for each minibatch
           for start_idx in range(0, train_sz, mini_batch_size):
               ...
               # for each element in the minibatch
               for i in range(start_idx, end_idx):
                   self.feedforward(x)
                   self.backprop(y)
               self.update_parameters(mini_batch_size, learning_rate)
   ```
   If we execute that in the device (XPU), the copy between cpu/device makes it worst.

Functions `torch_profile()` and `profile()` evidence this. Particularly, `torch_profile()`, running on 1000 samples with batch size 20, on the XPU:

| Name                  | Self CPU% | Self CPU  | Self XPU | Self XPU% | # of Calls |
|-----------------------|-----------|-----------|----------|-----------|------------|
| urEnqueueKernelLaunch | 39.03%    | 1.563s    | 0.000us  | 0.00%     | 43600      |
| aten::mm              | 22.52%    | 901.789ms | 5.928ms  | 17.93%    | 8000       |
| aten::add_            | 9.94%     | 398.136ms | 9.120ms  | 27.59%    | 12300      |

**Adding operations to the kernels queue (`urEnqueueKernelLaunch`) takes most of the time!** Also, the in-place addition (`add_`) from the accumulation takes an important proportion of time with respect to the tensor multiplications, roughly 44%.

Inefficient? Absolutely! But following versions will actually use batched computation, and (should) dramatically reduce the number of kernels queued, making the operations themselves larger, so they better take advantage of the XPU.

## Running the code
Place the `mnist.plk.gz` file at the root directory of this repo (you can get mnist [from here](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/refs/heads/master/data/mnist.pkl.gz)), activate the conda environment (see [System Setup](#system-setup) section), and then run the main file.

``` shell
$ ls mnist.pkl.gz
mnist.pkl.gz
$ conda activate torch-xpu
$ python main.py
```


## Changelog Summary
- This version:
  - Migrate tensors from NumPy to PyTorch.
  - Use XPU (Intel) for the first time!
  - Profile CPU and XPU activity.
  
- [Version 1.0.0](https://github.com/parra-ca/nn-from-scratch/tree/v1.0.0): 
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
