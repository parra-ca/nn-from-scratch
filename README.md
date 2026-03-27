# Neural Networks From Scratch to Framework

A learning project building a feedforward neural network from scratch, starting with manual NumPy implementation and progressively migrating it to a PyTorch training framework with data augmentation, regularization, and experiment tracking.

The whole framework consists of two files: `nn_framework.py` and `nn_driver.py`.

---

## Results

| Dataset  | Architecture               | Accuracy   |
|----------|----------------------------|------------|
| MNIST    | `[784, 784, 500, 100, 10]` | **99.12%** |
| CIFAR-10 | `[3072, 3000, 1000, 10]`   | **56.46%** |

MNIST result uses AdamW, cosine annealing LR schedule, dropout, and 16× Kornia augmentation (elastic deformation + affine transforms).
CIFAR-10 is a fully-connected architecture; a CNN would push this significantly higher.

### Effect of L2 regularization on small dataset (1000 samples).
![Weight decay](all_plots/06_weight-decay_plot.png)

### Effect of L2 regularization + dropout on small dataset.
![Weight decay + dropout](all_plots/07_weight-decay-dropout_plot.png)

### MNIST. Data Augmentation.
![MNIST augmentation](all_plots/08_mnist-augment_plot.png)

### CIFAR-10. Distributing 4000 neurons Architecture search (neuron distribution)
![CIFAR-10 neurons](all_plots/09_cifar10-neurons_plot.png)


---

## Progression

The implementation evolved across 9 iterations, each introducing one concept:

| File                                   | What was added                                                                                          |
|----------------------------------------|---------------------------------------------------------------------------------------------------------|
| `nn0--numpy__manual-implementation.py` | Manual forward/backward pass, pre-allocated buffers, pluggable function interface                       |
| `nn1--torch__use-tensors.py`           | NumPy -> PyTorch tensors, XPU targeting                                                                  |
| `nn2--torch__parallel-batches.py`      | Batched matrix operations                                                                               |
| `nn3--torch__autograd.py`              | Autograd replaces manual backprop                                                                       |
| `nn4--torch__modules.py`               | `nn.Module`, `nn.ModuleList`, parameter groups                                                          |
| `nn5--torch__cross-entropy.py`         | Cross-entropy loss replaces MSE                                                                         |
| `nn_driver.py + nn_framework.py`       | AdamW, weight decay, Cosine annealing LR scheduling, L2 regularization, Dropout, Checkpoint restoration, Plots, Parameters and hyperparameters export. |

Each step was validated empirically before moving to the next.

## Concepts Covered

Backpropagation, autograd, cross-entropy, SGD/AdamW, cosine annealing, L1/L2 regularization, dropout, data augmentation, float32 numerical stability tracing, optimizer state instrumentation.

---

## Framework

The final code is split into two files:

### `nn_framework.py` 
This is the reusable infrastructure with base classes and utility functions.
- `Loader` base class with `MnistLoader` and `CifarLoader` subclasses. It provides a unified data source interface.
- In-place shuffle using pre-allocated buffers (`Loader.data_train_alt_X`/`_Y`) to avoid new reallocation at each epoch.
- Kornia augmentation pipeline: elastic deformation, and affine transforms (rotate, shear, scale, shift)
- `Module` base class: Base class of the model. Xavier init, evaluate, checkpoint restoration, early termination (perfect score or collapse detection).
- JSON experiment logging and multi-run comparison plots.

### `nn_driver.py`
Experiment configuration and hyperparameters.
- Subclasses `Module` with a concrete model. You should implement the `forward`, `fit`, and maybe the `__init__` functions.
- All options and hyperparameters are stored in four plain dictionaries: `general`, `data_params`, `hyp_params`, `plot_params`
- For a new experiment, just create a new driver file.

### Adding a new dataset
- Create a new subclass of  `Loader` in `nn_framework.py`. Populate its `data_train`, `data_valid`, `data_test`, `img_dims`, `in_ly_sz`, `out_ly_sz`, attributes and the alt shuffle buffers `data_train_alt_X`/`_Y`.
- Then add a branch in `model_train_eval` so the driver can use the new loader based on some new string in `data_params["dataset"]`.

### Adding a new architecture

Subclass `Module`, override `forward` and `fit`. The base class still handles evaluation, checkpointing, accuracy logging, and parameter initialization.

---

## Setup

I am using an Intel Arc Pro B50 (on Fedora). If using CPU, or other graphic card, then you may want to setup things by yourself.

**Note:** Make sure your UEFI setting "Large BAR" is enabled. Intel says the option might be labeled "Re-Size BAR" or "Smart Access Memory" depending on your motherboard maker.

**Note 2:** As of February 2026, there was a bug that broke things with kernel 6.18. kernel 6.17 fixed it.

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
# Download...
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# and install conda
$ bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, allow init
source ~/.bashrc

# verify conda
conda --version

```


### Conda environment, and setup testing
```shell
# go to this repo's directory and check the conda environment.yml
$ cd path/to/this/repo

# create the conda environment
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

---

## Running

Datasets expected at:
- `datasets/mnist/mnist.pkl.gz`
- `datasets/cifar-10/` (standard CIFAR-10 binary format)

```python
# dictionaries with all the configuration
general = {...}
data_params = {...}
hyp_params = {...}
plot_params = {...}

# train and evaluate
nnf.model_train_eval(general, data_params, hyp_params, plot_params)

# plot all saved runs for a given experiment prefix
nnf.plot_all(plot_params, "09_cifar10-neurons/")
```

``` shell
$ ./nn_driver.py
```

Each training run saves a JSON file with hyperparameters and per-epoch accuracy.
`plot_all` reads all JSON files matching the prefix and overlays them in a single comparison plot. Parameters that are equal across all runs are show in the subtitle. The different ones appear in the legend of each plot.
