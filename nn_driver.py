#!/usr/bin/env python

from __future__ import annotations
from typing import Mapping, Any, override
import torch
from torch import Tensor, nn

# local imports
import nn_framework as nnf

# decision function
def argmax(a_batch: Tensor) -> Tensor:
    """Returns index of the highest logit in each sample of the received batch"""
    return torch.argmax(a_batch, dim=1, keepdim=True)
    
class NeuralNetwork(nnf.Module):
    """Feedforward netwrk with optional dropout, trained with AdamW
    and cosine annealing learning rate schedule."""
    @override
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
            model_fns: Mapping[str, Any],
            in_out_sz: tuple[int, int],
    ):
        super().__init__(hyp_params, model_fns, in_out_sz)
        if "dropout" in hyp_params:
            self.dropout = nn.Dropout(p=hyp_params["dropout"])
        return
    
    @override
    def forward(
            self,
            input_batch: Tensor,
    ) -> Tensor:
        # INPUT LAYER
        a = input_batch

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
            data_source: nnf.Loader,
            hyp_params: Mapping[str, Any],
    ) -> None:
        epochs = hyp_params["epochs"]
        batch_sz = hyp_params["batch_sz"]
        learning_rate = hyp_params["lrn_rate"]
        weight_decay = hyp_params["w_decay"]
        betas = hyp_params["betas"]

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
                X_batch,Y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]
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

if __name__ == "__main__":
    nnf.random_seed(10)
    general = {
        "title"     : "CIFAR-10: Distributing 4000 Neurons",
        "about"     : "AdamW. ReLU. Cosine Annealing LR sched.",
        # "prefix"    : "06_weight-decay/",
        # "prefix"    : "07_weight-decay-dropout/",
        # "prefix"    : "08_mnist-augment/",
        "prefix"    : "09_cifar10-neurons/",
    }
    data_params = {
        "dataset"     : "cifar-10",
        "device"      : "xpu",
        "factor"      : 1,
        "augm_params" : {},
        "export"      : 0,
        "exp_dir"     : "augm_samples",
    }
    hyp_params = {
        "epochs"      : 100,
        "batch_sz"    : 1000,
        "model"       : NeuralNetwork,
        "hid_layers"  : [3500, 250, 250],
        "load_path"   : None,
        "model_fns"   : {"activ_hl" : nn.functional.relu,
                         "loss"     : nn.functional.cross_entropy,
                         "decision" : argmax},
        "lrn_rate"    : 0.0005,
        "w_decay"     : 0.005,
        "betas"       : (0.9, 0.999),
        "dropout"     : 0.5,
    }
    plot_params = {
        "keys_general" : ["title", "about"],
        "keys_data"    : ["dataset", "data_sz", "factor"],
        "keys_hparam"  : ["epochs", "batch_sz", "hid_layers", "lrn_rate",
                          "w_decay", "betas", "dropout"],
        "size"         : (11,15),
        "x_range"      : "auto",
        # "y_range"      : (80, 95), # weight-decay
        # "y_range"      : (97,100), # mnist-augment
        "y_range"      : (45, 65), # cifar
        "plot_train"   : False,
        "plot_valid"   : True,
        "plot_test"    : True,
    }

    #nnf.model_train_eval(general, data_params, hyp_params, plot_params)
    nnf.plot_all(plot_params, general["prefix"])

