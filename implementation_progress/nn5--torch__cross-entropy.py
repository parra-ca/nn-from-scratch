from __future__ import annotations
from collections.abc import Callable
from typing import cast

import os, pickle, gzip

import torch
import torch.nn as nn
torch.set_printoptions(precision=3, sci_mode=False, linewidth=512)

# Activation functions
def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))
# Decision functions
def argmax(a: torch.Tensor) -> torch.Tensor: 
    return torch.argmax(a, dim=1, keepdim=True)


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

    def mini_sample(self, in_base=0, in_sz=784, train_sz=None, val_sz=None,
                    test_sz=None):
        train_sz = train_sz if train_sz is not None else self.tr_d[0].shape[0]
        val_sz = val_sz if val_sz is not None else self.va_d[0].shape[0]
        test_sz = test_sz if test_sz is not None else self.te_d[0].shape[0]

        self.tr_d = (self.tr_d[0][:train_sz,in_base:in_base+in_sz],
                     self.tr_d[1][:train_sz])
        self.va_d = (self.va_d[0][:val_sz,in_base:in_base+in_sz],
                     self.va_d[1][:val_sz])
        self.te_d = (self.te_d[0][:test_sz,in_base:in_base+in_sz],
                     self.te_d[1][:test_sz])
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
            activ_hl: Callable[[torch.Tensor], torch.Tensor],
            loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            decision: Callable[[torch.Tensor], torch.Tensor],
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
        self.fn_loss = loss
        self.fn_decision = decision
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
        return self.layers[-1](a)

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

        # train a number of epochs
        self.train()
        sgd = torch.optim.SGD(params=self.parameters(), lr=learning_rate)
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
                loss = self.fn_loss(out_ly_batch, Y_batch.squeeze().long())
                loss.backward()

                # update parameters
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


def main(
        data_path: str = "mnist.pkl.gz",
        epochs = 100,
        hidden_lys = [200, 200],
        batch_sz = 2000,
        lr = 0.5,
        do_test = True,
        load_path: str | None = None,
        save_path: str | None = None
): 
    print("\nPROBING SYSTEM")
    print("XPU available?:", torch.xpu.is_available())

    
    print("\nLOADING DATA")
    mnist = MnistLoader(data_path, device="cpu")
    #mnist.mini_sample(train_sz=1000, val_sz=1000, test_sz=1000)
    mnist.describe()

    
    print("\nCREATING NETWORK")
    in_ly_sz = [mnist.tr_d[0].shape[1]]
    out_ly_sz = [10]
    all_layers = in_ly_sz + hidden_lys + out_ly_sz
    print("in ->", all_layers, "-> out")
    my_net = NeuralNetwork(
        layer_sizes=all_layers,
        activ_hl=sigmoid,
        loss=nn.functional.cross_entropy,
        decision=argmax,
        seed=42
    )
    if load_path is not None:
        print(f"LOADING W+B FROM {load_path}")
        my_net.load_state_dict(torch.load(load_path, weights_only=True))
    #my_net.to("xpu")
    
    print(f"\nTRAINING\n"
          f"  epochs     : {epochs}\n"
          f"  batch sz   : {batch_sz}\n"
          f"  learn rate : {lr}")
    my_net.stoc_grad_descent(
        data_source=mnist,
        epochs=epochs,
        batch_sz=batch_sz,
        learning_rate=lr,
        test=do_test
    )

    if save_path is not None:
        print(f"SAVING W+B TO {save_path}")
        torch.save(my_net.state_dict(), save_path)
    print("DONE!")
    return

if __name__ == "__main__":
    try:
        script_name = os.path.basename(__file__)
        load_name = os.path.splitext(script_name)[0]+".pth"
        save_name = os.path.splitext(script_name)[0]+"_experiment.pth" 
        print(f"RUNNING: {script_name}")
        main(
            epochs=30,
            batch_sz=10,
            lr=0.5,
            hidden_lys=[30],
            do_test=True,
            #load_path=load_name,
            save_path=save_name
        )
    except KeyboardInterrupt:
        print("\n\n---- interrupted! ----\n")
        exit(0)

