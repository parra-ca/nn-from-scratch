from __future__ import annotations
from typing import Any
from torch import nn

from .. import models
from .. import dataloaders as dld
from .util import run_experiments 

data_override = {
    "sample"    : 0,
}
hyper_override = {
    "device"    : "xpu",
    "epochs"    : 200,
}
plot_override = {
    "size"      : (10,8),
    "x_range"   : "auto",
    "plot_train": True,
    "plot_valid": True,
    "plot_test" : True,
}

def fc_vs_cnn_on_mnist(
        do_train: bool = True,
        do_plot: bool = True
) -> None:
    general = {
        "title"       : "FC vs CNN (~1.06M parameters) on MNIST",
        "about"       : "AdamW. ReLU. Cosine Annealing LR sched.",
        "result_dir"  : "801_fc_vs_cnn_on_mnist",
        "seed"        : 10,
    }
    data_base = {
        "dataloader"  : dld.MnistLoader,
        "batch_sz"    : 80,
        "val_split"   : 0.1,
        "augm"        : True,
        "augm_params" : {"elastic": {"kernel_size": (57, 57),
                                     "sigma"      : (9.0, 9.0),
                                     "alpha"      : (15, 15),},
                         "affine" : {"degrees"    : (-10, 10),
                                     "shear"      : (-5, 5),
                                     "scale"      : (0.9, 1.1)},
                         "crop"   : {"size"       : (28, 28),
                                     "padding"    : 2}}
    }
    hyper_base = {
        "model"       : None,
        "model_fns"   : None,
        "load_path"   : None,
        "epochs"      : 200,
        "batch_sz"    : 80,
        "hid_layers"  : None,
        "dropout"     : 0.5,
        "lr"          : 0.0005,
        "lr_min"      : 1e-6,
        "w_decay"     : 0.0016,
        "betas"       : (0.9, 0.999),
    }
    plot_base = {
        "keys_general" : ["title", "about"],
        "keys_data"    : ["dataset", "augm"],
        "keys_hparam"  : None,
        "y_range"      : (90, 100),
    }
    variations = [
        [("h", "model", models.FullyConnectedNet),
         ("h", "model_fns", {"activ_hl" : nn.ReLU,
                             "loss"     : nn.functional.cross_entropy,
                             "decision" : models.common.argmax}),
         ("h", "hid_layers", [784,500,100]),
         ("p", "keys_hparam", ["epochs", "batch_sz", "hid_layers",
                               "dropout", "lr", "lr_min", "w_decay", "betas"])
        ],
        [("h", "model", models.ConvolutionalNet),
         ("h", "model_fns", {"conv_activ_hl" : nn.ReLU,
                             "head_activ_hl" : nn.ReLU,
                             "loss"          : nn.functional.cross_entropy,
                             "decision"      : models.common.argmax}),
         ("h", "conv_cksp", [
             ( 32,3,1,2), # ch_out, ker_sz, stride, mxpool_sz
             ( 64,3,1,2)]),
         ("h", "to_head", "flatten"),
         ("h", "head_hid_lys", [320, 100]),
         ("p", "keys_hparam", ["epochs", "batch_sz", "conv_cksp", "head_hid_lys",
                               "dropout", "lr", "lr_min", "w_decay", "betas"])
         ]
    ]
    
    run_experiments(hyper_base, hyper_override, data_base, data_override,
                    plot_base, plot_override, general, variations,
                    do_train, do_plot)
    return

def fc_vs_cnn_on_cifar(
        do_train: bool = True,
        do_plot: bool = True
) -> None:
    general = {
        "title"       : "FC vs CNN (~4.2M parameters) on CIFAR-10",
        "about"       : "AdamW. ReLU. Cosine Annealing LR sched.",
        "result_dir"  : "802_fc_vs_cnn_on_cifar",
        "seed"        : 10,
    }
    data_base = {
        "dataloader"  : dld.CifarLoader,
        "batch_sz"    : 80,
        "val_split"   : 0.1,
        "augm"        : True,
        "augm_params" : {# "elastic": {"kernel_size": (57, 57),
                         #             "sigma"      : (9.0, 9.0),
                         #             "alpha"      : (15, 15),},
                         "hflip"  : {},
                         # "affine" : {"degrees"    : (-10, 10),
                         #             "shear"      : (-5, 5),
                         #             "scale"      : (0.9, 1.1)},
                         "color"  : {"brightness" : 0.2,
                                     "contrast"   : 0.2,
                                     "saturation" : 0.2,
                                     "hue"        : 0.05,},
                         "crop"   : {"size"       : (32, 32),
                                     "padding"    : 2}}
    }
    hyper_base = {
        "model"       : None,
        "model_fns"   : None,
        "load_path"   : None,
        "epochs"      : 200,
        "batch_sz"    : 80,
        "hid_layers"  : None,
        "dropout"     : 0.5,
        "lr"          : 0.0005,
        "lr_min"      : 1e-6,
        "w_decay"     : 0.0016,
        "betas"       : (0.9, 0.999),
    }
    plot_base = {
        "keys_general" : ["title", "about"],
        "keys_data"    : ["dataset", "augm"],
        "keys_hparam"  : None,
        "y_range"      : (30, 100),
    }
    variations = [
        [("h", "model", models.FullyConnectedNet),
         ("h", "model_fns", {"activ_hl" : nn.ReLU,
                             "loss"     : nn.functional.cross_entropy,
                             "decision" : models.common.argmax}),
         ("h", "hid_layers", [3500, 1195]),
         ("p", "keys_hparam", ["epochs", "batch_sz", "hid_layers",
                               "dropout", "lr", "lr_min", "w_decay", "betas"])
        ],
        [("h", "model", models.ConvolutionalNet),
         ("h", "model_fns", {"conv_activ_hl" : nn.ReLU,
                             "head_activ_hl" : nn.ReLU,
                             "loss"          : nn.functional.cross_entropy,
                             "decision"      : models.common.argmax}),
         ("h", "conv_cksp", [
             (128,3,1,2), # ch_out, ker_sz, stride, mxpool_sz
             (256,3,1,2),
             (512,3,1,2),
             (512,3,1,2)]),
         ("h", "to_head", "gap"),
         ("h", "head_hid_lys", [500, 200]),
         ("p", "keys_hparam", ["epochs", "batch_sz", "conv_cksp", "head_hid_lys",
                               "dropout", "lr", "lr_min", "w_decay", "betas"])
         ]
    ]
    run_experiments(hyper_base, hyper_override, data_base, data_override,
                    plot_base, plot_override, general, variations,
                    do_train, do_plot)
    return

