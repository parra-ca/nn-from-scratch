from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import pickle
import gzip

import math
import torch
import torch.nn as nn
torch.set_printoptions(precision=3, sci_mode=False, linewidth=512)

class FnActivation(Protocol):
    def __call__(self, z: torch.Tensor) -> torch.Tensor: ...
class FnLoss(Protocol):
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
class FnDecision(Protocol):
    def __call__(self, a: torch.Tensor) -> torch.Tensor: ...
class FnDecisionInv(Protocol):
    def __call__(self, y: torch.Tensor, out_layer: torch.Tensor) -> torch.Tensor: ...
# Activation functions
def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))
# Cost and cost derivative functions
def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = y_hat - y
    return 0.5 * torch.sum(diff * diff)
# Decision functions
def argmax(a: torch.Tensor) -> torch.Tensor: 
    return torch.argmax(a, dim=1, keepdim=True)
def argmax_inv(y: torch.Tensor, out_layer: torch.Tensor):
    out = torch.zeros_like(out_layer)
    return torch.scatter(out, dim=1, index=y.long(), value=1.0)


class MnistLoader(object):
    def __init__(self, path, device="xpu"):
        f = gzip.open(path, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.tr_d, self.va_d, self.te_d = u.load()
        f.close()
        self.flatten_data(torch.device(device))

    def flatten_data(self, device):
        for dname in ("tr_d", "va_d", "te_d"):
            X,Y = getattr(self, dname)
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            Y_tensor = torch.tensor(Y, dtype=torch.int32, device=device).reshape(-1,1)
            setattr(self, dname, (X_tensor, Y_tensor))

        # alternative memory blocks for data shuffling
        self.tr_d_alt_X = torch.empty_like(self.tr_d[0])
        self.tr_d_alt_Y = torch.empty_like(self.tr_d[1])
        return

    def shuffle_training_data(self):
        # tr_d --> shuffle --> tr_d_alt
        X, Y = self.tr_d
        rand_indices = torch.randperm(X.shape[0], device=X.device)
        torch.index_select(X, dim=0, index=rand_indices, out=self.tr_d_alt_X)
        torch.index_select(Y, dim=0, index=rand_indices, out=self.tr_d_alt_Y)

        # flip pointers so that tr_d has the shuffled data
        tmp_tr = self.tr_d
        self.tr_d = (self.tr_d_alt_X, self.tr_d_alt_Y)
        self.tr_d_alt_X = tmp_tr[0]
        self.tr_d_alt_Y = tmp_tr[1]
        return

    def mini_sample(self):
        in_base = 0
        in_size = 784
        ds_size = self.tr_d[0].shape[0]
        self.tr_d = (self.tr_d[0][:ds_size,in_base:in_base+in_size],
                     self.tr_d[1][:ds_size])
        self.te_d = (self.te_d[0][:ds_size,in_base:in_base+in_size],
                     self.te_d[1][:ds_size])
        self.va_d = (self.va_d[0][:ds_size,in_base:in_base+in_size],
                     self.va_d[1][:ds_size])
        self.tr_d_alt_X = torch.empty_like(self.tr_d[0])
        self.tr_d_alt_Y = torch.empty_like(self.tr_d[1])
        return
    
    def describe(self):
        names = ("Training", "Validation", "Test")
        parts = (self.tr_d, self.va_d, self.te_d)
        for name, data_set in zip(names, parts):
            x,y = data_set
            xdt, ydt = str(x.dtype), str(y.dtype)
            print(f"{name}:(X,Y):\n"
                  f"    X   {xdt:>14}   {list(x.shape)}\n"
                  f"    Y   {ydt:>14}   {list(y.shape)}")
        return
    
class NeuralNetwork(nn.Module):
    def __init__(
            self,
            layer_sizes: list[int],
            activ_hl: FnActivation,
            activ_ol: FnActivation,
            loss: FnLoss,
            decision: FnDecision,
            decision_inv: FnDecisionInv,
            seed=42
    ) -> None:
        super().__init__()
        self.__random_seed(seed)

        # list of layers
        self.layers = nn.ModuleList(
            [nn.Linear(fan_in, fan_out, bias=True)
             for fan_in,fan_out in zip(layer_sizes[:-1],layer_sizes[1:])]
        )

        # initial parameters with Xavier (and zero for biases)
        for l in self.layers:
            l = cast(nn.Linear, l)
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            
        # hidden and output activation functions
        self.fn_activ_hl = activ_hl
        self.fn_activ_ol = activ_ol
        self.fn_loss = loss
        self.fn_decision = decision
        self.fn_decision_inv = decision_inv
        return
    
    @staticmethod
    def __random_seed(
            seed: int = 42) -> None:
        torch.manual_seed(seed);        
        if torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return
    
    def forward(
            self,
            input_batch: torch.Tensor
    ) -> torch.Tensor:
        # INPUT LAYER
        a = input_batch

        # HIDDEN LAYERS
        for layer in self.layers[:-1]:
            a = self.fn_activ_hl(layer(a))

        # OUTPUT LAYER
        return self.fn_activ_ol(self.layers[-1](a))

    def stoc_grad_descent(
            self,
            data_source: MnistLoader,
            epochs: int,
            batch_sz: int,
            learning_rate: float,
            test: bool = False
    ):
        # trim train data to fit an exact number of batches
        train_full_sz = data_source.tr_d[0].shape[0]  
        train_sz = train_full_sz - (train_full_sz % batch_sz)
        alpha = learning_rate / batch_sz

        # train a number of epochs
        self.train()
        sgd = torch.optim.SGD(params=self.parameters(), lr=alpha)
        for epoch in range(epochs):
            print(f"Epoch {(epoch+1):3}: ", end="", flush=True)

            # shuffle data before creating the mini-batches
            if epoch != 0:
                data_source.shuffle_training_data()

            # do one gradient step per mini_batch
            X,Y = data_source.tr_d
            for start_idx in range(0, train_sz, batch_sz):
                end_idx = start_idx+batch_sz
                X_batch,Y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]

                # clear gradients
                sgd.zero_grad()

                # forward and backward pass
                out_ly_batch = self(X_batch)
                y_as_ly_batch = self.fn_decision_inv(Y_batch, out_ly_batch)
                loss = self.fn_loss(out_ly_batch, y_as_ly_batch)
                loss.backward()

                sgd.step()

            
            # test how good the network is doing
            if (test or epoch+1 == epochs) and data_source.te_d is not None:
                test_data = data_source.te_d
                correct, total = self._evaluate(test_data, batch_sz) 
                perc = 100 * correct / total
                print(f"{perc:6.2f}%")
                if abs(perc - 100) < 0.0000001:
                    print("Good enough! Early exit.")
                    break
                self.train()
            else:
                print(f"complete.")

    def _evaluate(
            self,
            test_data: tuple[torch.Tensor, torch.Tensor],
            batch_sz: int
    ) -> tuple[int, int]:

        # trim train data to fit an exact number of batches
        test_full_sz = test_data[0].shape[0]
        test_sz = test_full_sz - (test_full_sz % batch_sz)

        # evaluate in batches
        self.eval()
        with torch.no_grad():
            X,Y = test_data
            correct_count = torch.zeros((1,), dtype=torch.int64, device=Y.device)
            for start_idx in range(0, test_sz, batch_sz):
                end_idx = start_idx+batch_sz
                X_batch,Y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]
                
                out_layer = self(X_batch)
                y_hat_batch = self.fn_decision(out_layer)
                correct_count += torch.sum(y_hat_batch == Y_batch)

            # return the total correct predictions and the total samples
            corr_count_int = int(correct_count.item())
        return (corr_count_int, test_sz)


def main(data_path: str = "mnist.pkl.gz"):
    print("PROBING SYSTEM")
    print("XPU available?:", torch.xpu.is_available())

    
    print("\nLOADING DATA")
    mnist = MnistLoader(data_path)
    mnist.describe()

    
    print("\nCREATING NETWORK")
    in_ly_sz = [mnist.tr_d[0].shape[1]]
    hidden_lys = [200, 200]
    out_ly_sz = [10]
    all_layers = in_ly_sz + hidden_lys + out_ly_sz
    print("in ->", all_layers, "-> out")
    my_net = NeuralNetwork(
        all_layers,
        sigmoid, sigmoid, mse, argmax, argmax_inv,
        42
    )
    my_net.to("xpu")
        
    epochs = 100
    batch_size = 20
    learning_rate = 0.5
    do_test = True
    print(f"\nTRAINING\n"
          f"  epochs    : {epochs}\n"
          f"  batch sz  : {batch_size}\n"
          f"  learn rate: {learning_rate}")
    my_net.stoc_grad_descent(
        data_source=mnist,
        epochs=epochs,
        batch_sz=batch_size,
        learning_rate=learning_rate,
        test=do_test
    )

    print("DONE!")
    return

if __name__ == "__main__":
    main()
