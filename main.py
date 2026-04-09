from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import pickle
import gzip
_MINI_DATASET_SIZE = 100

import math
import torch
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

funcs = {"hl_activ": sigmoid, "ol_activ": sigmoid, "loss": mse,
         "decision": argmax, "decision_inv": argmax_inv}


class MnistLoader(object):
    def __init__(self, path, device="xpu"):
        try:
            f = gzip.open(path, 'rb')
        except FileNotFoundError as e:
            print(f"Could not find '{path}'.")
            exit(1)
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
        ds_size = _MINI_DATASET_SIZE
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
    
class NeuralNetwork(object):
    def __init__(
            self,
            weights: list[torch.Tensor],
            biases: list[torch.Tensor],
            funcs: Mapping[str, Callable[..., Any]],
            dtype: torch.dtype,
            device: str | torch.device
    ) -> None:
        # general network attributes
        self.dtype = dtype
        self.device = torch.device(device)
        
        # Check correct weights and biases, and store in object
        net_parts = {"weights":weights, "biases":biases}
        for name,lst in net_parts.items():
            if not isinstance(lst, list):
                raise TypeError(f"{name} is not a list, got {type(lst)}.")
            for i,item in enumerate(lst):
                if not isinstance(item, torch.Tensor):
                    raise TypeError(f"{name}[{i}] is not a torch.Tensor, "
                                    f"got {type(item)}.")
                if item.dtype != self.dtype:
                    raise TypeError(f"{name}[{i}] dtype is not {self.dtype}, "
                                    f"got {item.dtype}.")
                if item.device != self.device:
                    lst[i] = item.to(self.device)
        for i in range(len(weights)):
            ws0 = weights[i].shape[1]
            bs0 = biases[i].shape[1]
            if ws0 != bs0:
                raise ValueError(f"Error: weights[{i}].shape[1] == {ws0} != "
                                 f"biases[{i}].shape[1] == {bs0}")
        self.weights = weights
        self.biases = biases

        # user defined activation, loss, decision, and equal functions
        self.fn_hl_activ: FnActivation
        self.fn_ol_activ: FnActivation
        self.fn_loss: FnLoss
        self.fn_decision: FnDecision
        self.fn_decision_inv: FnDecisionInv
        keys = ["hl_activ", "ol_activ", "loss", "decision", "decision_inv"]
        for k in keys:
            if k not in funcs:
                raise ValueError(f"Function \"{k}\" expected but not given.")
            fn = funcs[k]
            if not callable(fn):
                raise ValueError(f"funcs[\"{k}\"] is not callable.")
            setattr(self, f'fn_{k}', fn)
        return
    
    @classmethod
    def from_weights(
            cls,
            init_weights: list,
            init_biases: list,
            funcs: Mapping[str, Callable[..., Any]],
            dtype: torch.dtype = torch.float32,
            rand_seed: int = 42,
            device: str | torch.device = "xpu"
    ):
        device = torch.device(device)
        cls.__random_seed(device, rand_seed)
        
        if len(init_weights) != len(init_biases):
            raise ValueError(f"len(weights) != len(biases)")
        # weights (prepend a dummy zero to keep easy indices)
        W_list = [torch.as_tensor(w, dtype=dtype, device=device)\
                  .requires_grad_(True)
                  for w in init_weights]
        weights = [torch.empty((0,0), dtype=dtype, device=device)] + W_list
        # biases
        b_list = [torch.as_tensor(b, dtype=dtype, device=device)\
                  .reshape(1,-1)\
                  .requires_grad_(True)
                  for b in init_biases]
        biases = [torch.empty((0,0), dtype=dtype, device=device)] + b_list
        return cls(weights, biases, funcs, dtype, device)
    
    @classmethod
    def from_layers(
            cls,
            ly_sizes: list,
            funcs: Mapping[str, Callable[..., Any]],
            dtype: torch.dtype = torch.float32,
            rand_seed: int = 42,
            device: str | torch.device = "xpu"
    ):
        device = torch.device(device)
        cls.__random_seed(device, rand_seed)
        
        # weights and biases
        weights = [torch.empty((0,0), dtype=dtype, device=device)]
        biases = [torch.empty((0,0), dtype=dtype, device=device)]
        for l_sz, ll_sz in zip(ly_sizes[:-1], ly_sizes[1:]):
            xavier = 1/math.sqrt(l_sz)
            w = torch.empty((l_sz,ll_sz), dtype=dtype, device=device)\
                     .uniform_(-xavier, xavier)\
                     .requires_grad_(True)
            b = torch.empty((1,ll_sz), dtype=dtype, device=device)\
                     .uniform_(-xavier, xavier)\
                     .requires_grad_(True)
            weights.append(w)
            biases.append(b)
        return cls(weights, biases, funcs, dtype, device)
    
    @classmethod
    def __random_seed(
            cls,
            device: str | torch.device = "xpu",
            seed: int = 42) -> None:
        old = getattr(cls, "seed", None)
        if old is not None and old != seed:
            raise Exception("seed_torch can only be called once.")
        cls.seed = seed
        torch.manual_seed(seed);
        if torch.device(device).type == "xpu":
            torch.xpu.manual_seed_all(seed)
        return
    
    def _feedforward(self, a: torch.Tensor) -> torch.Tensor:
        # INPUT LAYER: the first 'a' is the input batch

        # HIDDEN LAYERS: z_l = a_{l-1} times W_l ; a_l = activ(z_l)
        for l in range(1, len(self.weights)-1):
            z = a @ self.weights[l] + self.biases[l]
            a = self.fn_hl_activ(z)

        # OUTPUT LAYER: same as hidden layers, but it may have a different
        # activation function
        z = a @ self.weights[-1] + self.biases[-1]
        return self.fn_ol_activ(z)

    def _evaluate(
            self,
            test_data: tuple[torch.Tensor, torch.Tensor],
            batch_sz: int
    ) -> tuple[int, int]:

        # trim train data to fit an exact number of batches
        test_full_sz = test_data[0].shape[0]
        test_sz = test_full_sz - (test_full_sz % batch_sz)

        # evaluate in batches
        with torch.no_grad():
            X,Y = test_data
            correct_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
            for start_idx in range(0, test_sz, batch_sz):
                end_idx = start_idx+batch_sz

                out_layer = self._feedforward(X[start_idx:end_idx,:])
                y_hat_batch = self.fn_decision(out_layer)
                y_batch = Y[start_idx:end_idx,:]
                correct_count += torch.sum(y_hat_batch == y_batch)

            # return the total correct predictions and the total samples
            corr_count_int = int(correct_count.item())
        return (corr_count_int, test_sz)
    
    def stoc_gradient_descent(
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

                # clear all bias and weight gradients
                for w,b in zip(self.weights[1:], self.biases[1:]):
                    if w.grad is not None:
                        w.grad.zero_()
                    if b.grad is not None:
                        b.grad.zero_()
                
                # forward and backwards pass
                out_layer_batch = self._feedforward(X_batch)
                y_as_layer_batch = self.fn_decision_inv(Y_batch,
                                                        out_layer_batch)
                loss = self.fn_loss(out_layer_batch, y_as_layer_batch) 
                loss.backward()

                # update parameters
                with torch.no_grad():
                    alpha = learning_rate / batch_sz
                    for w,b in zip(self.weights[1:], self.biases[1:]):
                        assert w.grad is not None
                        assert b.grad is not None
                        w -= alpha * w.grad
                        b -= alpha * b.grad
            
            # test how good the network is doing
            if (test or epoch+1 == epochs) and data_source.te_d is not None:
                test_data = data_source.te_d
                correct, total = self._evaluate(test_data, batch_sz) 
                perc = 100 * correct / total
                print(f"{perc:6.2f}%")
                if abs(perc - 100) < 0.0000001:
                    print("Good enough! Early exit.")
                    break
            else:
                print(f"complete.")

        return

    def save_params(self, path:str):
        payload = {
            "weights": [w.detach().cpu() for w in self.weights[1:]],
            "biases":  [b.detach().cpu() for b in self.biases[1:]]
        }
        torch.save(payload, path)
        return
    
    @classmethod
    def load_params(cls, path:str):
        payload = torch.load(path, weights_only=True)
        return payload["weights"], payload["biases"]

    def describe(self, print_values=False):
        s = []
        w_parts = ["\nWEIGHTS:"]
        for i, w in enumerate(self.weights):
            ob = w if print_values else ''
            w_parts.append(f'  L{i} {list(w.shape)}\n  {ob}')
        w_parts = "\n".join(w_parts)
        s.append(w_parts)
        
        b_parts = ["\nBIASES:"]
        for i,b in enumerate(self.biases):
            ob = b if print_values else ''
            b_parts.append(f'  L{i} {list(b.shape)}\n  {ob}')
        b_parts = "\n".join(b_parts)
        s.append(b_parts)
        
        return "\n".join(s)
        
    def __str__(self):
        return self.describe(print_values=False)

def run_sample(nn: NeuralNetwork, train_data, batch_sz: int, samples: int = 100):
    X, Y = train_data
    for start_idx in range(0, samples, batch_sz):
        end_idx = min(samples, start_idx+batch_sz)
        X_batch,Y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]
        out_layer_batch = nn._feedforward(X_batch)
        y_as_layer_batch = nn.fn_decision_inv(Y_batch, out_layer_batch)
        loss = nn.fn_loss(out_layer_batch, y_as_layer_batch)
        loss.backward()
        with torch.no_grad():
            alpha = 0.5 / batch_sz
            for w,b in zip(nn.weights[1:], nn.biases[1:]):
                assert w.grad is not None
                assert b.grad is not None
                w -= alpha * w.grad
                b -= alpha * b.grad
    return

def profile(nn: NeuralNetwork,
            data_source: MnistLoader,
            batch_size: int,
            samples: int):
    print("PROFILING")
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    run_sample(nn, data_source.tr_d, batch_size, samples)
    profiler.disable()
    profiler.print_stats(sort="tottime")
    exit(0)

def torch_profile(nn: NeuralNetwork,
                  data_source: MnistLoader,
                  batch_size: int,
                  samples: int = 100):
    print("PROFILING")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ],
        acc_events=True
    ) as prof:
        run_sample(nn, data_source.tr_d, batch_size, samples)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    exit(0)

def main(data_path: str = "mnist.pkl.gz",
         params_path: str | None = "mnist_params",
         profile: bool = False):
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
    nn = NeuralNetwork.from_layers(all_layers, funcs)

    
    epochs = 100
    batch_size = 20
    learning_rate = 1
    do_test = True
    print(f"\nTRAINING\n"
          f"  epochs    : {epochs}\n"
          f"  batch sz  : {batch_size}\n"
          f"  learn rate: {learning_rate}")
    if profile:
        torch_profile(
            nn,
            data_source=mnist,
            batch_size=batch_size,
            samples=1000
        )

    nn.stoc_gradient_descent(
        data_source=mnist,
        epochs=epochs,
        batch_sz=batch_size,
        learning_rate=learning_rate,
        test=do_test
    )
    
    if params_path:
        print("\nSAVE PARAMETERS")
        nn.save_params(params_path)

    print("DONE!")
    return

if __name__ == "__main__":
    main(params_path=None, profile=False)
