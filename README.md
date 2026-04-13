# Neural Networks, From Scratch to Framework

A learning project that intends to evolve from a simple neural network built from scratch, to a PyTorch based framework.

1. [Results](#results)
2. [Running the code](#running-the-code)
3. [Changelog Summary](#changelog-summary)
4. [System Setup](#system-setup)



## Results
| Dataset  | Fully Connected | Convolutional |
|----------|-----------------|---------------|
| MNIST    | **99.14%**      | **99.39%**    |
| CIFAR-10 | **51.50%**      | **90.59%**    |

### Fully Connected vs Convolutional models
Convolutional networks take input images  with a given spatial size (image width/height) and a few channels (3 in RGB images). Typically, as the data goes trough the convolutions, the spatial size is reduced while new channels with more conceptual meanings are created. A fully connected "head" interprets the last activation map to take the final decision.

Unlike fully connected networks, this architecture share weights across neurons of the same layer in the "kernel" patch. It also preserves spatial information that makes them much more suitable for image recognition tasks. The following plot shows a comparison between two networks with roughly the same number of parameters on CIFAR-10. One is Fully connected (blue) and the other is Convolutional (orange). The improvement is massive!
![Fully Connected vs Convolutional on CIFAR-10 Dataset](img/802_fc_vs_cnn_on_cifar_plot.png)

MNIST is an easier dataset that reaches good accuracy with a fully connected network, however, a convolutional network of a similar number of parameters still shows some improvement.
![Fully Connected vs Convolutional on MNIST Dataset](img/801_fc_vs_cnn_on_mnist_plot.png)

### Data augmentation
Train data can be augmented by introducing small alterations to the original train dataset. Below, 9 input samples of MNIST and CIFAR-10 with some random alterations at their right. Made with the `kornia` python module.
Different augmentations make sense for different scenarios: in gray-scale character recognition you don't want to flip the image or alter the colors, but in photos, that is a reasonable augmentation.
![Augmented MNIST Data](img/mnist_augmented.png)
![Augmented CIFAR-10 Data](img/cifar_augmented.png)

The following plot shows the training of a fully connected network on MNIST, using AdamW with Cosine Annealing LR scheduling. Using only the original dataset of 50K samples yields a 98.66% accuracy (blue line). Augmenting the data improves accuracy; particularly, when augmented 16x, a 99.25% accuracy is obtained.
![Augmented Training on MNIST](img/711_mnist_augmentation_plot.png)

Here is the same fully connected architecture training on CIFAR-10. The original dataset (50K - 5K validation samples) shows 48.98% test accuracy. A 16x data augmentation improves it to 56.31%. Although a convolutional model is a much better architecture for image classification, artificially augmenting the training data is still a powerful tool.
![Augmented Training on CIFAR-10](img/721_cifar10_augmentation_plot.png)

### Weight decay and Dropout
Regularization techniques, in general, discourage complex models that rely on specific features, activations, or weights; in an attempt to avoid memorizing the training data (overfitting) and actually produce a model that is more general.

In order to overfit the training data and see the effects of regularization, let's trim it to 5000 samples. With no weight decay or dropout (blue), training accuracy quickly  reaches 100% (thin line), but validation accuracy (thicker line) peak at 94.67% (star marker); resulting in the final test accuracy of 94.13% (triangle marker).

Each, weight decay and dropout help to reduce the gap between train and validation accuracy. But combined, they make the model generalize even better.

![Weight Decay](img/701_SGD_weight-decay_plot.png)
![Dropout](img/702_SGD_dropout_plot.png)
![Weight Decay + Dropout](img/703_SGD_weight-decay-and-dropout_plot.png)

## Running the code
Activate the conda environment (see [System Setup](#system-setup) section), and then run the main file as a python module. Add the cli arguments `train` and `plot` to perform those actions.

``` shell
$ conda activate torch-xpu
$ python -m nnfw.main train plot

ACCELERATOR: XPU

LOADING DATA
Dataset           : cifar-10 (60000 samples)
Online augmenting : True
Training:
    samples  : 45000
    batch_sz : 80
    features : [3, 32, 32] torch.float32 
    classes  : 10          <class 'int'> 
Validation:
    samples  : 5000
    batch_sz : 1024
    features : [3, 32, 32] torch.float32 
    classes  : 10          <class 'int'> 
Testing:
    samples  : 10000
    batch_sz : 1024
    features : [3, 32, 32] torch.float32 
    classes  : 10          <class 'int'> 

CREATING MODEL
  model        : ConvolutionalNet
  epochs       : 200
  batch_sz     : 80
  dropout      : 0.5
  lr           : 0.0005
  lr_min       : 1e-06
  w_decay      : 0.0016
  betas        : (0.9, 0.999)
  conv_cksp    : [(128, 3, 1, 2), (256, 3, 1, 2), (512, 3, 1, 2), (512, 3, 1, 2)]
  head_hid_lys : [500, 200]

TRAINING
Epoch   1: tra: 55.66%   val: 52.88% <-- new best!
Epoch   2: tra: 63.62%   val: 62.26% <-- new best!
Epoch   3: tra: 71.42%   val: 68.66% <-- new best!
Epoch   4: tra: 72.18%   val: 70.72% <-- new best!
Epoch   5: tra: 77.42%   val: 74.60% <-- new best!
Epoch   6: tra: 77.38%   val: 74.26%
Epoch   7: tra: 79.34%   val: 76.24% <-- new best!
Epoch   8: tra: 81.80%   val: 78.42% <-- new best!
...
Epoch 167: tra: 99.98%   val: 90.82% <-- new best!
...
Epoch 192: tra:100.00%   val: 90.26%
Epoch 193: tra:100.00%   val: 90.56%
Epoch 194: tra:100.00%   val: 90.50%
Epoch 195: tra:100.00%   val: 90.34%
Epoch 196: tra:100.00%   val: 90.34%
Epoch 197: tra:100.00%   val: 90.52%
Epoch 198: tra:100.00%   val: 90.36%
Epoch 199: tra:100.00%   val: 90.50%
Epoch 200: tra:100.00%   val: 90.54%

SAVING PARAMS: results/802_fc_vs_cnn_on_cifar/090.59__0412-212148.params

SAVING PLOT DATA: results/802_fc_vs_cnn_on_cifar/090.59__0412-212148.json

PLOTTING ALL JSON FILES IN 'results/802_fc_vs_cnn_on_cifar'.
```

To add more experiments, add new function definitions in [`nnfw/experiments/experiments.py`](nnfw/experiments/experiments.py), and add that function to the `list_of_experiments` in [`nnfw/main.py`](nnfw/main.py)

## Changelog Summary
- Version 8.0.0:
  - Replace static data augmentation with online augmentation (per batch) with no extra memory cost.
  - Refactor code separating: data loaders, models, plotting, and experiments.
    ```text
    nnfw/
    ├── dataloaders/
    │   └── common.py
    ├── experiments/
    │   ├── util.py
    │   └── experiments.py
    ├── main.py
    ├── models/
    │   ├── cnn.py
    │   ├── common.py
    │   └── fc.py
    ├── plotter.py
    └── util.py
    ```
  - Add convolutional model in [`nnfw/models/cnn.py`](nnfw/models/cnn.py)
  - Concentrate all experiments parameters in [`nnfw/experiments/experiments.py`](nnfw/experiments/experiments.py)
  - Use PyTorch Data Loaders rather than manually handling the datasets from the `tar` files.

- [Version 7.0.0](../../tree/v7.0.0):
  - Split code into `main.py` and `nnfw.py` (Neural Network framework)
  
  - **Activation function, regularization, and optimization loop**:
    - Replaced sigmoid activation for ReLU (faster math, and does not saturate, while still introducing non-linearity)
    - Add Weight decay only for weights (not for biases), and Dorpout.
    - Learning Rate Scheduling (Cosine Annealing).
    - Replace stochastic gradient descent with AdamW.

  - **Data augmentation and sample exporting**:
    - Add manual data augmentation though Kornia library. Including randomized: elastic deformations, corp, horizontal flip, affine transformation, and color variations.
    - Specify a multiplier factor to augment (or trim) the original training data, which is shuffled and kept as the training dataset. (Validation and test data never augmented or shuffled.) This makes each augmentation be visited once per epoch.
    - Augmentation is performed once, and results are kept in-memory, avoiding online computation at the expense of memory.
    - `Loader._export_img()`: Original and Augmented images are exported and drawn side by side.
    
  - **Generalization of Data Loader and Model**:
     - Generalize `Loader` class for common Loader interface and add `MnistLoader` and `CifarLoader` ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)) subclasses.
    - Generalize neural network model class `Module(nn.Module)` so subclasses only need to implement an updated `__init__()`, `forward()` and `fit()`.

  - **Training monitoring and checkpoint**:
    - Monitor validation accuracy, tracking best performing set of parameters (checkpoint); and applying final parameters restoration.
    - Add network collapse detection.

  - **Training plots**:
    - Add training, validation, and testing plot production.
    - Save training log data (metadata and train/validation/test accuracy values) for multi-training comparative training (in JSON format)
    - Multi-train plots group common hyperparameters into the plot's title and add only contrasting values to legend.

- [Version 6.0.0](../../tree/v6.0.0):
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

That should be it! You can test your setup with this tiny python script:
``` python
import torch
print("torch         :", torch.__version__)
print("xpu available :", torch.xpu.is_available())
print("device        :", torch.xpu.get_device_name(0))
print("mem_get_info  :", torch.xpu.mem_get_info())

x = torch.randn(1, device="xpu")
torch.xpu.synchronize()
y = x.cpu()
print("x.cpu().item() =", y.item())
```

The output should look something like this (the last number is random)
``` text
torch         : 2.10.0+xpu
xpu available : True
device        : Intel(R) Graphics [0xe212]
mem_get_info  : (9370324992, 16241180672)
x.cpu().item() = 2.148054838180542
```
