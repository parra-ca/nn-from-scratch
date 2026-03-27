from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any

import pickle
import gzip

import math
import torch

# Activation and activation prime functions
def sigmoid(z: torch.Tensor, out: torch.Tensor):
    # out = 1 / (1 + exp(-z))
    torch.negative(z, out=out)
    out.exp_()
    out.add_(1.0)
    out.reciprocal_()
    return
def sigmoid_prime_from_sigm(a: torch.Tensor, z: None, out: torch.Tensor):
    # out = a * (1-a)
    torch.sub(1.0, a, out=out)
    out.mul_(a)
    return
def linear(z: torch.Tensor, out: torch.Tensor):
    # out = z
    out.copy_(z)
    return
def linear_prime(a: torch.Tensor, z: None, out: torch.Tensor):
    # out = 1
    out.fill_(1.0)
    return

# Decision and decision inverse functions
def flatten(a: torch.Tensor) -> torch.Tensor: # previously called "identity"
    return a.ravel()
def argmax(a: torch.Tensor) -> torch.Tensor: 
    return torch.argmax(a)
def argmax_inv(y: torch.Tensor, out: torch.Tensor) -> None:
    out.fill_(0.0)
    out[y] = 1.0
    return
# Equality tests functions
def equal_float(y_hat: torch.Tensor, y: torch.Tensor) -> bool:
    return torch.allclose(y_hat, y, atol=1e-6)
def equal_int(y_hat: torch.Tensor, y: torch.Tensor) -> bool:
    return torch.equal(y_hat, y)

# Cost and cost derivative functions
def mse(y_hat: torch.Tensor, y: torch.Tensor,
        tmp: torch.Tensor) -> torch.Tensor:
    # return = (1/2) * Sum((y_hat - y)^2)
    torch.sub(y_hat, y, out=tmp)
    tmp.mul_(tmp)
    return 0.5 * tmp.sum()
def mse_prime(y_hat: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    # out = y_hat - y
    torch.sub(y_hat, y, out=out)
    return

funcs = {
    "hl_activ": sigmoid,
    "hl_activ_prime": sigmoid_prime_from_sigm,
    "ol_activ": sigmoid,
    "ol_activ_prime": sigmoid_prime_from_sigm,
    "loss": mse,
    "loss_prime": mse_prime,
    "decision": argmax,
    "decision_inv": argmax_inv,
    "equal": equal_int
}


class MnistLoader(object):
    def __init__(self, path, device="xpu"):
        f = gzip.open(path, 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.tr_d, self.va_d, self.te_d = u.load()
        f.close()
        self.flatten_data(device)

    def flatten_data(self, device):
        for dname in ("tr_d", "va_d", "te_d"):
            X,Y = getattr(self, dname)
            X_t = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in X])
            X_t = X_t.reshape(X_t.shape[0], -1, 1).to(device=device)
            Y_t = torch.as_tensor(Y, dtype=torch.int16, device=device)
            setattr(self, dname, list(zip(X_t, Y_t)))
        return

    def describe(self):
        names = ("Training", "Validation", "Test")
        parts = (self.tr_d, self.va_d, self.te_d)
        for name, part in zip(names, parts):
            x,y = part[0]
            print(f"{name:10} ({len(part)} samples):\n"
                  f"    X:({x.dtype}){x.shape} --> y:({y.dtype}){y.shape}")
        return

class NeuralNetwork(object):
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
    
    def __init__(
            self,
            weights: list[torch.Tensor],
            biases: list[torch.Tensor],
            layers: list[torch.Tensor],
            funcs: Mapping[str, Callable[..., Any]],
            dtype: torch.dtype,
            device: str | torch.device
    ) -> None:
        # general network attributes
        self.dtype = dtype
        self.device = torch.device(device)

        # network shape
        for i,ly in enumerate(layers):
            if ly.ndim != 2 or ly.shape[1] != 1:
                raise ValueError(f"layers[{i}].shape != (n,1), got {ly.shape}.")
        net_parts = {"weights":weights,
                     "biases":biases,
                     "layers":layers}
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
        self.weights = weights
        self.biases = biases
        self.layers = layers
        self.ly_sizes = [ly.shape[0] for ly in self.layers]

        # gradient accumulators
        self.weights_acc_grad = [torch.zeros_like(w) for w in weights]
        self.biases_acc_grad = [torch.zeros_like(b) for b in biases]

        # user defined activation, loss, decision, and equal functions
        keys = ["hl_activ", "hl_activ_prime",
                "ol_activ", "ol_activ_prime",
                "loss", "loss_prime",
                "decision", "decision_inv",
                "equal"]
        for k in keys:
            if k not in funcs:
                raise ValueError(f"Function \"{k}\" expected but not given.")
            fn = funcs[k]
            if not callable(fn):
                raise ValueError(f"funcs[\"{k}\"] is not callable.")
            setattr(self, f'fn_{k}', fn)

        # scratch space for temporary values
        max_ly_sz = max(self.ly_sizes)
        max_W_sz = max(ly*nxt_ly for ly, nxt_ly in
                       zip(self.ly_sizes[:-1], self.ly_sizes[1:]))
        # to hold the activation prime
        self.tmp_a_prime = torch.zeros((max_ly_sz,), dtype=self.dtype,
                                       device=self.device)
        # to hold the activation gradient
        self.tmp_a_grad  = torch.zeros((max_ly_sz,), dtype=self.dtype,
                                       device=self.device)
        # to hold the delta of a layer's weights
        self.tmp_w_grad = torch.zeros((max_W_sz,), dtype=self.dtype,
                                      device=self.device)
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
        
        # weights
        W_list = [torch.as_tensor(w, dtype=dtype, device=device)
                  for w in init_weights]
        weights = [torch.empty((0,), dtype=dtype, device=device)] + W_list

        # biases
        b_list = [torch.as_tensor(b, dtype=dtype, device=device).reshape(-1,1)
                  for b in init_biases]
        biases = [torch.empty((0,), dtype=dtype, device=device)] + b_list
        
        # list of layer sizes
        ly_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]

        # layers
        layers = [torch.zeros((l_sz,1), dtype=dtype, device=device)
                  for l_sz in ly_sizes]
        
        # check sizes
        for i in range(1, len(weights)):
            if weights[i].shape[0] != biases[i].shape[0]:
                raise ValueError(f"weights[{i}]={str(weights[i].shape)} "
                                 f"does not match bias[{i}]="
                                 f"{str(biases[i].shape)}.")
            
        return cls(weights, biases, layers, funcs, dtype, device)
    
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
        
        # layers
        layers = [torch.zeros((ly_sz,1), dtype=dtype, device=device) for ly_sz in ly_sizes]
        
        # weights and biases
        weights = [torch.empty((0,), dtype=dtype, device=device)]
        biases = [torch.empty((0,), dtype=dtype, device=device)]
        for l_sz, ll_sz in zip(ly_sizes[:-1], ly_sizes[1:]):
            xavier = 1/math.sqrt(l_sz)
            w = torch.empty((ll_sz,l_sz), dtype=dtype, device=device)\
                     .uniform_(-xavier, xavier)
            b = torch.empty((ll_sz,1), dtype=dtype, device=device)\
                     .uniform_(-xavier, xavier)
            weights.append(w)
            biases.append(b)
        return cls(weights, biases, layers, funcs, dtype, device)

    
    # These 'fn_' functions are to be replaced in __init__ with the
    # ones provided by the user
    def fn_hl_activ(self, z: torch.Tensor, out: torch.Tensor):
        # activation function of hidden layers
        return
    def fn_hl_activ_prime(self, a: torch.Tensor, z: torch.Tensor | None, out: torch.Tensor):
        # derivative of activation function of hidden layers
        return
    def fn_ol_activ(self, z: torch.Tensor, out: torch.Tensor):
        # activation function of output layer
        return
    def fn_ol_activ_prime(self, a: torch.Tensor, z: torch.Tensor | None, out: torch.Tensor):
        # derivative of activation function of output layer
        return
    def fn_decision(self, a: torch.Tensor) -> torch.Tensor:
        # interprets the last layer into the actual y_hat
        return []
    def fn_decision_inv(self, y: torch.Tensor , out: torch.Tensor) -> None:
        # expresses y as the expected values of the last layer
        return y
    def fn_equal(self, y_hat: torch.Tensor, y: torch.Tensor) -> bool:
        # how to determine if they_hat is equal to y
        return False
    def fn_loss(self, y_hat: torch.Tensor, y: torch.Tensor, tmp: torch.Tensor) -> float:
        return 0.0
    def fn_loss_prime(self, y_hat: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
        # gradient of C with respect to output layer
        return

    def feedforward(self, x: torch.Tensor):
        # INPUT LAYER: simply assign the input data to this layer
        self.layers[0].copy_(x)

        # HIDDEN LAYERS: z_l = W_l times a_{l-1}; a_l = activ(z_l)
        for l in range(1, len(self.ly_sizes)-1):
            torch.matmul(self.weights[l], self.layers[l-1], out=self.layers[l])
            self.layers[l].add_(self.biases[l])
            self.fn_hl_activ(self.layers[l], self.layers[l])

        # OUTPUT LAYER: same as hidden layers, but it may have a different
        # activation function
        torch.matmul(self.weights[-1], self.layers[-2], out=self.layers[-1])
        self.layers[-1].add_(self.biases[-1])
        self.fn_ol_activ(self.layers[-1], self.layers[-1])
        return
    
    def backprop(self, y: torch.Tensor):
        # Use the same space that now have the activations to store
        # the deltas.

        # STARTUP:
        # - compute delta_z of output layer
        out_layer = self.layers[-1]
        ol_sz = out_layer.shape[0]
        ol_a_prime = self.tmp_a_prime[:ol_sz].reshape(-1,1)
        ol_a_grad  = self.tmp_a_grad[:ol_sz].reshape(-1,1)
        y_as_layer = ol_a_grad 
        self.fn_decision_inv(y=y, out=y_as_layer)
        self.fn_ol_activ_prime(a=out_layer, z=None, out=ol_a_prime)
        self.fn_loss_prime(y_hat=out_layer, y=y_as_layer, out=ol_a_grad)
        torch.mul(ol_a_prime, ol_a_grad, out=out_layer)

        # OUTPUT AND ALL BUT FIRST HIDDEN LAYERS:
        # - compute gradient of C w.r.t. this layer (tl) weights and biases.
        # - compute delta_z of previous layer (pl) (only hidden layers).
        lys = len(self.layers)
        for pl,tl in zip(range(lys-2, -1, -1), range(lys-1, 0, -1)):
            prev_layer = self.layers[pl]
            pl_sz = prev_layer.shape[0]
            #this_layer = self.layers[tl]
            tl_sz = self.layers[tl].shape[0]
            
            # gradient of C w.r.t. this layer biases
            tl_delta_z = self.layers[tl]
            acc_b_grad = self.biases_acc_grad[tl]
            acc_b_grad.add_(tl_delta_z)
            
            # gradient of C w.r.t. this layer weights
            tl_w_grad = self.tmp_w_grad[:tl_sz*pl_sz].reshape(tl_sz,pl_sz)
            acc_w_grad = self.weights_acc_grad[tl]
            torch.matmul(tl_delta_z, prev_layer.T, out=tl_w_grad)
            acc_w_grad.add_(tl_w_grad)

            if pl > 0: # delta_z of previous *hidden* layers only
                pl_a_prime = self.tmp_a_prime[:pl_sz].reshape(-1,1)
                pl_a_grad = self.tmp_a_grad[:pl_sz].reshape(-1,1)
                tl_w = self.weights[tl]
                self.fn_hl_activ_prime(a=prev_layer, z=None, out=pl_a_prime)
                torch.matmul(tl_w.T, tl_delta_z, out=pl_a_grad)
                torch.mul(pl_a_prime, pl_a_grad, out=prev_layer)
        return
    
    def update_parameters(self, mini_batch_sz: int, learning_rate: float):
        learn_factor = - learning_rate / mini_batch_sz
        # self.{weights,biases,..._acc_grad}[0] are dummy.
        for ly in range(1, len(self.weights)):
            b = self.biases[ly]
            b_gr = self.biases_acc_grad[ly]
            b_gr.mul_(learn_factor)
            b.add_(b_gr)

            w = self.weights[ly]
            w_gr = self.weights_acc_grad[ly]
            w_gr.mul_(learn_factor)
            w.add_(w_gr)
        return
    
    def evaluate(
            self,
            test_data: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> int:
        correct_count = torch.zeros((1,), dtype=torch.int64,
                                     device=self.device)
        for x,y in test_data:
            self.feedforward(x)
            out_layer = self.layers[-1]
            y_hat = self.fn_decision(out_layer)
            if self.fn_equal(y_hat, y):
                correct_count.add_(1)
        return int(correct_count.item())

    def stoc_gradient_descent(
            self,
            train_data: list[tuple[torch.Tensor,torch.Tensor]],
            epochs: int,
            mini_batch_size: int,
            learning_rate: float,
            test_data: list[tuple[torch.Tensor,torch.Tensor]] | None = None
    ):
        train_sz = len(train_data)
        for epoch in range(epochs):
            print(f"Epoch {(epoch+1):3}: ", end="", flush=True)
            # randomize samples order
            rand_idx = torch.randperm(train_sz)
            train_data = [train_data[i] for i in rand_idx]
            
            # do one descent step for each mini-batch
            for start_idx in range(0, train_sz, mini_batch_size):
                end_idx = min(start_idx+mini_batch_size, train_sz)
                # clear all bias and weight gradiant accumulators
                for b_gr,w_gr in zip(self.biases_acc_grad[1:],
                                     self.weights_acc_grad[1:]):
                    b_gr.fill_(0.0)
                    w_gr.fill_(0.0)
                # Do forward and backward passes for all samples
                # in the mini_batch, accumulating the gradients
                # produced by each sample.
                for i in range(start_idx, end_idx):
                    # read a randomized sample
                    x,y = train_data[i]
                    self.feedforward(x)
                    self.backprop(y)
                self.update_parameters(mini_batch_size, learning_rate)

            # test how good the network is doing
            if test_data:
                test_sz = len(test_data)
                perc = 100 * self.evaluate(test_data) / test_sz
                print(f"{perc:6.2f}%")
                if abs(perc - 100) < 0.0000001:
                    print("good enough!")
                    break
            else:
                print(f"complete.")
        return

    def save_params(self, path:str):
        payload = {
            "weights": [w.cpu() for w in self.weights[1:]],
            "biases":  [b.cpu() for b in self.biases[1:]]
        }
        torch.save(payload, path)
        return
    
    @classmethod
    def load_params(cls, path:str):
        payload = torch.load(path, weights_only=True)
        return payload["weights"], payload["biases"]
    
    def __str__(self):
        s = []

        l_parts = []
        for l in self.layers:
            l_parts.append(f'{l}\n')
        l_parts = "\n".join(l_parts)

        w_parts = []
        for w in self.weights:
            w_parts.append(f'{w}\n')
        w_parts = "\n".join(w_parts)
        
        s.extend(["LAYERS:", l_parts])
        s.extend(["WEIGHTS:", w_parts])
        
        return "\n".join(s)

    def __repr__(self):
        s = []
        
        w_g_parts = []
        for wg in self.weights_acc_grad:
            w_g_parts.append(f'{wg}\n')
        w_g_parts = "\n".join(w_g_parts)
        
        b_g_parts = []
        for bg in self.biases_acc_grad:
            b_g_parts.append(f'{bg}\n')
        b_g_parts = "\n".join(b_g_parts)

        t_parts = ["tmp_a_prime:", str(self.tmp_a_prime),
                   "tmp_a_grad:", str(self.tmp_a_grad),
                   "tmp_w_grad:", str(self.tmp_w_grad)]
        t_parts = "\n".join(t_parts)
        
        s.extend(["WEIGHTS GRADIENTS:", w_g_parts])
        s.extend(["BIAS GRADIENTS:", b_g_parts])
        s.extend(["TEMP_MEMORY:", t_parts])
        
        return self.__str__()+"\n"+"\n".join(s)

def profile_network(nn: NeuralNetwork, train_data, n_samples: int = 100):
    print("PROFILING")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ]
    ) as prof:
        for i in range(n_samples):
            x, y = train_data[i]
            nn.feedforward(x)
            nn.backprop(y)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    exit(0)


def main(data_path="mnist.pkl.gz", params_path="mnist_params"):
    print("PROBING SYSTEM")
    print("XPU available?:", torch.xpu.is_available())

    print("\nLOADING DATA")
    mnist = MnistLoader(data_path)
    mnist.describe()
    
    print("\nCREATING NETWORK")
    in_ly_sz = [len(mnist.tr_d[0][0])]
    hidden_lys = [200,200]
    out_ly_sz = [10]
    all_layers = in_ly_sz + hidden_lys + out_ly_sz
    print("in ->", all_layers, "-> out")
    nn = NeuralNetwork.from_layers(all_layers, funcs)

    # profile_network(nn, data.tr_d)

    print("\nTRAINING")
    nn.stoc_gradient_descent(
        train_data=mnist.tr_d,
        epochs=100,
        mini_batch_size=20,
        learning_rate=0.5,
        test_data=mnist.te_d
        )

    if params_path:
        print("\nSAVE PARAMETERS")
        nn.save_params(params_path)

    print("DONE!")
    return

if __name__ == "__main__":
    main(params_path=None)
