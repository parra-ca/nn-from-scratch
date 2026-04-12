from __future__ import annotations
from typing import Mapping, Any, cast, override
import torch
from torch import Tensor, nn
import nnfw

def argmax(a_batch: Tensor) -> Tensor:
    """Returns index of the highest logit in each sample of the received batch"""
    return torch.argmax(a_batch, dim=1, keepdim=True)
    
class NeuralNetwork(nnfw.Module):
    """Feedforward netwrk with optional dropout, trained with AdamW
    and cosine annealing learning rate schedule."""
    @override
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
            model_fns: Mapping[str, Any],
            data_source: nnfw.Loader,
    ):
        super().__init__(hyp_params, model_fns, data_source)

        # list of layers
        hid_layers = hyp_params["hid_layers"]
        layer_sizes = [data_source.in_ly_sz] + hid_layers + \
            [data_source.out_ly_sz]
        self.layers = nn.ModuleList(
            [nn.Linear(fan_in, fan_out, bias=True, dtype=nnfw.FLOAT_DTYPE,
                       device=self.device)
             for fan_in,fan_out in zip(layer_sizes[:-1],layer_sizes[1:])]
        )

        if "dropout" in hyp_params:
            self.dropout = nn.Dropout(p=hyp_params["dropout"])

        # initial parameters with Xavier (and zero for biases)
        for l in self.layers:
            l = cast(nn.Linear, l)
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        return
    
    @override
    def forward(
            self,
            input_batch: Tensor,
    ) -> Tensor:
        # INPUT LAYER
        a = input_batch.flatten(start_dim=1)

        # HIDDEN LAYERS
        for layer in self.layers[:-1]:
            a = self.fn_activ_hl(layer(a))

            # apply dropout if it was part of the net construction.
            if hasattr(self, "dropout"):
                a = self.dropout(a)
            
        # OUTPUT LAYER
        return self.layers[-1](a)

    @override
    def fit(
            self,
            data_source: nnfw.Loader,
    ) -> None:
        epochs = self.hyp_params["epochs"]
        batch_sz = self.hyp_params["batch_sz"]
        learning_rate = self.hyp_params["lrn_rate"]
        weight_decay = self.hyp_params["w_decay"]
        betas = self.hyp_params["betas"]

        # trim train data to fit an exact number of batches
        train_full_sz = data_source.data_train[0].shape[0]  
        train_sz = train_full_sz - (train_full_sz % batch_sz)

        # extract unshuffled subset of training set to monitor accuracy
        data_sz = data_source.data_valid[0].shape[0]
        train_sample = (data_source.data_train[0][:data_sz].clone(),
                        data_source.data_train[1][:data_sz].clone())

        # create optimizer
        self.train()
        optimizer = torch.optim.AdamW(
            params=[
                {"params":[p for n,p in self.named_parameters() if 'weight' in n],
                 "weight_decay": weight_decay},
                {"params":[p for n,p in self.named_parameters() if 'bias' in n],
                 "weight_decay": 0.0}
            ],
            lr=learning_rate,
            betas=betas,
        )

        # create scheduled learn rate
        sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
        
        # train a number of epochs
        for epoch in range(epochs):
            print(f"Epoch {(epoch+1):3}: ", end="", flush=True)
            # shuffle data before creating the mini-batches
            if epoch != 0:
                data_source.shuffle_training_data()

            # do one gradient step per mini_batch
            X,Y = data_source.data_train
            for start_idx in range(0, train_sz, batch_sz):
                end_idx = start_idx+batch_sz
                X_batch = X[start_idx:end_idx].to(self.device)
                Y_batch = Y[start_idx:end_idx].to(self.device)
                optimizer.zero_grad()
                out_ly_batch = self(X_batch)
                loss = self.fn_loss(out_ly_batch, Y_batch.squeeze().long())
                loss.backward()
                optimizer.step()

            # update learning rate
            sched_lr.step()
            
            # monitor network accuracy
            accu_msg, do_break = self.monitor_accuracy(
                train_sample, data_source.data_valid, batch_sz, save_best=True)
            print(accu_msg)
            if do_break:
                break

        # restore parameters of the best performing net
        self.eval()
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)
        else:
            raise RuntimeError("No best state to restore")
        return

def exp__weight_decay_dropout(general, data_params, hyp_params, plot_params,
                              just_plot=False):
    general = dict(**general)
    data_params = dict(**data_params)
    hyp_params = dict(**hyp_params)
    plot_params = dict(**plot_params)
    general["about"] = "SGD. ReLU. Cosine Annealing LR sched."
    pars = [
        (
            "Regularization on small dataset: Weight Decay",
            "plots/701_SGD_weight-decay_",
            [100, 100],
            100,
            [(0.0000, 0.00), (0.0005, 0.00), (0.0010, 0.00)]
         ),(
            "Regularization on small dataset: Dropout",
            "plots/702_SGD_dropout_",
            [100, 100],
            100,
            [(0.0000, 0.00), (0.0000, 0.25), (0.0000, 0.50)],
        ),(
            "Regularization on small dataset: Weight Decay + Dropout",
            "plots/703_SGD_weight-decay-and-dropout_",
            [100, 100],
            100,
            [(0.0000, 0.00), (0.0005, 0.20)]
        ),
    ]
    plot_params["y_range"] = (93, 100)

    for ti,pf,ly,bs,wd_dr in pars:
        general["title"] = ti
        general["prefix"] = pf
        hyp_params["hid_layers"] = ly
        hyp_params["batch_sz"] = bs
        for wd,dr in wd_dr:
            nnfw.random_seed(10)
            hyp_params["w_decay"] = wd
            hyp_params["dropout"] = dr
            if just_plot:
                nnfw.plot_all(plot_params, general["prefix"])
                break
            nnfw.model_train_eval(general, data_params, hyp_params, plot_params)
            nnfw.plot_all(plot_params, general["prefix"])
    return

def exp__augment_mnist(general, data_params, hyp_params, plot_params,
                       just_plot=False):
    general = dict(**general)
    data_params = dict(**data_params)
    hyp_params = dict(**hyp_params)
    plot_params = dict(**plot_params)
    general["title"] = "MNIST Data Augmentation"
    general["about"] = "AdamW. ReLU. Cosine Annealing LR sched."
    general["prefix"] = "plots/711_mnist_augmentation_"
    hyp_params["hid_layers"] = [784, 500, 100]
    data_params["augm_params"] = {
        "elastic": {"kernel_size": (57, 57),
                    "sigma"      : (9.0, 9.0),
                    "alpha"      : (15, 15),},
        #"hflip"  : {},
        "affine" : {"degrees"    : (-10, 10),
                    "shear"      : (-5, 5),
                    "scale"      : (0.9, 1.1)},
        "crop"   : {"size"       : (28, 28),
                    "padding"    : 2},
        # "color"  : {"brightness" : 0.2,
        #             "contrast"   : 0.2,
        #             "saturation" : 0.2,
        #             "hue"        : 0.05,},
    }
    plot_params["y_range"] = (96, 100)
    factors = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0]
    for f in factors:
        nnfw.random_seed(10)
        data_params["factor"] = f
        if just_plot:
            nnfw.plot_all(plot_params, general["prefix"])
            break
        nnfw.model_train_eval(general, data_params, hyp_params, plot_params)
        nnfw.plot_all(plot_params, general["prefix"])
    return

def exp__augment_cifar(general, data_params, hyp_params, plot_params,
                       just_plot=False):
    general = dict(**general)
    data_params = dict(**data_params)
    hyp_params = dict(**hyp_params)
    plot_params = dict(**plot_params)
    general["title"] = "CIFAR-10 Data Augmentation"
    general["about"] = "AdamW. ReLU. Cosine Annealing LR sched."
    general["prefix"] = "plots/721_cifar10_augmentation_"
    hyp_params["hid_layers"] = [3000, 1000]
    data_params["dataset"] = "cifar-10"
    plot_params["y_range"] = (40, 65)
    data_params["augm_params"] = {
        # "elastic": {"kernel_size": (57, 57),
        #             "sigma"      : (9.0, 9.0),
        #             "alpha"      : (15, 15),},
        "hflip"  : {},
        # "affine" : {"degrees"    : (-10, 10),
        #             "shear"      : (-5, 5),
        #             "scale"      : (0.9, 1.1)},
        "crop"   : {"size"       : (32, 32),
                    "padding"    : 2},
        "color"  : {"brightness" : 0.2,
                    "contrast"   : 0.2,
                    "saturation" : 0.2,
                    "hue"        : 0.05,},
    }

    factors = [1.0, 2.0, 4.0, 8.0, 16.0]
    for f in factors:
        nnfw.random_seed(10)
        data_params["factor"] = f
        if just_plot:
            nnfw.plot_all(plot_params, general["prefix"])
            break
        nnfw.model_train_eval(general, data_params, hyp_params, plot_params)
        nnfw.plot_all(plot_params, general["prefix"])
    return

if __name__ == "__main__":
    general = {
        "title"     : "---",
        "about"     : "---",
        "prefix"    : "---",
    }
    data_params = {
        "dataset"     : "mnist",
        "factor"      : 1,
        "augm_params" : {},
        "export"      : 0,
        "exp_dir"     : "augm_samples",
    }
    hyp_params = {
        "device"      : "xpu",
        "epochs"      : 200,
        "batch_sz"    : 80,
        "model"       : NeuralNetwork,
        "hid_layers"  : [200, 200],
        "load_path"   : None,
        "model_fns"   : {"activ_hl" : nn.functional.relu,
                         "loss"     : nn.functional.cross_entropy,
                         "decision" : argmax},
        "lrn_rate"    : 0.0005,
        "w_decay"     : 0.0016,
        "betas"       : (0.9, 0.999),
        "dropout"     : 0.5,
    }
    plot_params = {
        "keys_general" : ["title", "about"],
        "keys_data"    : ["dataset", "train_sz"],
        "keys_hparam"  : ["epochs", "batch_sz", "hid_layers", "lrn_rate",
                          "w_decay", "dropout"],
        "size"         : (10,8),
        "x_range"      : "auto",
        "y_range"      : (96, 100),
        "plot_train"   : True,
        "plot_valid"   : True,
        "plot_test"    : True,
    }

    exp__weight_decay_dropout(general, data_params, hyp_params, plot_params,
                              just_plot=False)
    exp__augment_mnist(general, data_params, hyp_params, plot_params,
                       just_plot=False)
    exp__augment_cifar(general, data_params, hyp_params, plot_params,
                       just_plot=False)

    
