from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any

import pickle
import gzip

import math
import torch

torch.set_printoptions(precision=3, sci_mode=False, linewidth=512)

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
def flatten(a: torch.Tensor) -> torch.Tensor:
    return a.ravel()
def argmax(a: torch.Tensor) -> torch.Tensor: 
    return torch.argmax(a, dim=1, keepdim=True)
def argmax_inv(y: torch.Tensor, out: torch.Tensor):
    out.zero_()
    out.scatter_(dim=1, index=y, value=1.0)
    return
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
}


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

class Workspace(object):
    def __init__(
            self,
            weights: list[torch.Tensor],
            batch_size: int,
            dtype: torch.dtype,
            device: str | torch.device
    ):
        device = torch.device(device)
        self.batch_sz = batch_size
        
        # size of layers. Consider that the first weight is dummy.
        lys_sz = [weights[1].shape[0]] + [w.shape[1] for w in weights[1:]]
        max_ly_sz = max(lys_sz)
        
        # scratch space to hold intermediate computation:
        # - layers. Can contain activation (not counting input layer) or
        #   delta_z (again, not counting the input layer)
        self.layers = [torch.zeros((batch_size,ly_sz), dtype=dtype,
                                   device=device)
                       for ly_sz in lys_sz]
        
        # - single layer activation prime
        # - single layer activation gradient
        self.a_prime = torch.zeros((batch_size,max_ly_sz), dtype=dtype,
                                   device=device)
        self.a_grad  = torch.zeros((batch_size,max_ly_sz), dtype=dtype,
                                   device=device)
        
        # - gradient of Cost wrt all weights
        # - gradient of Cost wrt all biases
        self.w_grad = [torch.zeros((0,0), dtype=dtype, device=device)] + \
            [torch.zeros(w.shape, dtype=dtype, device=device)
             for w in weights[1:]]
        self.b_grad = [torch.zeros((0,0), dtype=dtype, device=device)] + \
            [torch.zeros((1,w.shape[1]), dtype=dtype, device=device)
             for w in weights[1:]]

        # simple accumulator for the evaluation count
        self.correct_count = torch.zeros(1, dtype=torch.int64, device=device)
        return
    
    def drop(self):
        attrs = ["batch_sz", "layers", "a_prime", "a_grad", "w_grad", "b_grad"]
        for a in attrs:
            if hasattr(self, a):
                delattr(self, a)
        torch.xpu.empty_cache()
        return

    def drop_train_space(self):
        attrs = ["batch_sz", "a_prime", "a_grad", "w_grad", "b_grad"]
        for a in attrs:
            if hasattr(self, a):
                delattr(self, a)
        torch.xpu.empty_cache()
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
            funcs: Mapping[str, Callable[..., Any]],
            dtype: torch.dtype,
            device: str | torch.device
    ) -> None:
        # general network attributes
        self.dtype = dtype
        self.device = torch.device(device)
        self.ws: Workspace | None = None
        
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
        keys = ["hl_activ", "hl_activ_prime",
                "ol_activ", "ol_activ_prime",
                "loss", "loss_prime",
                "decision", "decision_inv"]
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
        W_list = [torch.as_tensor(w, dtype=dtype, device=device)
                  for w in init_weights]
        weights = [torch.empty((0,0), dtype=dtype, device=device)] + W_list
        # biases
        b_list = [torch.as_tensor(b, dtype=dtype, device=device).reshape(1,-1)
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
                     .uniform_(-xavier, xavier)
            b = torch.empty((1,ll_sz), dtype=dtype, device=device)\
                     .uniform_(-xavier, xavier)
            weights.append(w)
            biases.append(b)
        return cls(weights, biases, funcs, dtype, device)
    
    # These 'fn_' functions are to be replaced in __init__ with the
    # ones provided by the user
    def fn_hl_activ(self, z: torch.Tensor, out: torch.Tensor):
        # activation function of hidden layers
        return
    def fn_hl_activ_prime(self, a: torch.Tensor, z: torch.Tensor | None,
                          out: torch.Tensor):
        # derivative of activation function of hidden layers
        return
    def fn_ol_activ(self, z: torch.Tensor, out: torch.Tensor):
        # activation function of output layer
        return
    def fn_ol_activ_prime(self, a: torch.Tensor, z: torch.Tensor | None,
                          out: torch.Tensor):
        # derivative of activation function of output layer
        return
    def fn_decision(self, a: torch.Tensor) -> torch.Tensor:
        # interprets the last layer into the actual y_hat
        return []
    def fn_decision_inv(self, y: torch.Tensor , out: torch.Tensor) -> None:
        # expresses y as the expected values of the last layer
        return y
    def fn_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                tmp: torch.Tensor) -> float:
        return 0.0
    def fn_loss_prime(self, y_hat: torch.Tensor, y: torch.Tensor,
                      out: torch.Tensor):
        # gradient of C with respect to output layer
        return

    def _allocate_train_workspace(self, batch_size: int):
        if self.ws is not None:
            self.ws.drop()
        self.ws = Workspace(self.weights, batch_size, self.dtype, self.device)
        return
        
    def _allocate_prediction_workspace(self):
        if self.ws is not None:
            self.ws.drop()
        self.ws = Workspace(self.weights, 1, self.dtype, self.device)
        self.ws.drop_train_space()
        return
    
    def _feedforward(self):
        # assumes the first layer already has the input data in place
        assert(self.ws is not None)
        
        # HIDDEN LAYERS: z_l = a_{l-1} times W_l ; a_l = activ(z_l)
        for l in range(1, len(self.weights)-1):
            torch.matmul(self.ws.layers[l-1],
                         self.weights[l],
                         out=self.ws.layers[l])
            self.ws.layers[l].add_(self.biases[l])
            self.fn_hl_activ(self.ws.layers[l], self.ws.layers[l])

        # OUTPUT LAYER: same as hidden layers, but it may have a different
        # activation function
        torch.matmul(self.ws.layers[-2],
                     self.weights[-1],
                     out=self.ws.layers[-1])
        self.ws.layers[-1].add_(self.biases[-1])
        self.fn_ol_activ(self.ws.layers[-1], self.ws.layers[-1])
        return

    def _backprop(self, y_batch: torch.Tensor):
        assert(self.ws is not None)
        batch_sz = y_batch.shape[0]
        # STARTUP:
        # - compute delta_z of output layer
        out_layer = self.ws.layers[-1]
        ly_sz = out_layer.shape[1]
        a_prime = self.ws.a_prime.ravel()[:batch_sz*ly_sz]\
                                 .reshape(batch_sz,ly_sz)
        a_grad  = self.ws.a_grad.ravel()[:batch_sz*ly_sz]\
                                .reshape(batch_sz,ly_sz)
        y_as_layer = a_grad
        self.fn_decision_inv(y=y_batch, out=y_as_layer)
        self.fn_loss_prime(y_hat=out_layer, y=y_as_layer, out=a_grad)
        self.fn_ol_activ_prime(a=out_layer, z=None, out=a_prime)
        torch.mul(a_grad, a_prime, out=out_layer)

        # OUTPUT AND ALL PREVIOUS, BUT FIRST HIDDEN LAYER:
        # - compute gradient of C w.r.t. this layer (tl) weights and biases.
        # - compute delta_z of previous layer (pl) (only hidden layers).
        lys = len(self.ws.layers)
        for pl,tl in zip(range(lys-2, -1, -1), range(lys-1, 0, -1)):
            # previous layer activation
            pl_activ = self.ws.layers[pl]
            pl_activ_sz = pl_activ.shape[1]
            # this layer delta
            tl_delta = self.ws.layers[tl]
            
            # gradient of C w.r.t. this layer biases:
            # - sum of deltas across batch samples
            tl_b_grad = self.ws.b_grad[tl]
            torch.sum(tl_delta, dim=0, keepdim=True, out=tl_b_grad)
            
            # gradient of C w.r.t. this layer weights:
            # - matrix multip. of this layer's delta and transposed previous
            #   layer activations.
            tl_w_grad = self.ws.w_grad[tl]
            torch.matmul(pl_activ.T, tl_delta, out=tl_w_grad)

            if pl > 0: # compute delta_z for previous layer only if it is hidden
                pl_a_prime = self.ws.a_prime.ravel()[:batch_sz*pl_activ_sz]\
                                            .reshape(batch_sz,pl_activ_sz)
                pl_a_grad = self.ws.a_grad.ravel()[:batch_sz*pl_activ_sz]\
                                          .reshape(batch_sz,pl_activ_sz)
                tl_w = self.weights[tl]
                self.fn_hl_activ_prime(a=pl_activ, z=None, out=pl_a_prime)
                torch.matmul(tl_delta, tl_w.T, out=pl_a_grad)
                torch.mul(pl_a_grad, pl_a_prime, out=pl_activ)
        return
    
    def _update_parameters(self, batch_sz: int, learning_rate: float):
        assert(self.ws is not None)
        learn_factor = - learning_rate / batch_sz
        # index 0 is dummy
        for ly in range(1, len(self.weights)):
            # update this layer's biases
            b = self.biases[ly]
            b_gr = self.ws.b_grad[ly]
            b_gr.mul_(learn_factor)
            b.add_(b_gr)

            # update this layer's weights
            w = self.weights[ly]
            w_gr = self.ws.w_grad[ly]
            w_gr.mul_(learn_factor)
            w.add_(w_gr)
        return

    def _evaluate(
            self,
            test_data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[int, int]:
        assert(self.ws is not None)
        # trim train data to fit an exact number of batches
        test_full_sz = test_data[0].shape[0]  
        batch_sz = self.ws.batch_sz
        test_sz = test_full_sz - (test_full_sz % batch_sz)

        # evaluate in batches
        X,Y = test_data
        correct_count = self.ws.correct_count
        for start_idx in range(0, test_sz, batch_sz):
            end_idx = start_idx+batch_sz

            self.ws.layers[0].copy_(X[start_idx:end_idx,:])
            self._feedforward()
            y_hat_batch = self.fn_decision(self.ws.layers[-1])
            y_batch = Y[start_idx:end_idx,:]
            correct_count += torch.sum(y_hat_batch == y_batch, )
        corr_count = int(correct_count.item())
        correct_count.zero_()
        return (corr_count, test_sz)
    
    def stoc_gradient_descent(
            self,
            data_source: MnistLoader,
            epochs: int,
            batch_sz: int,
            learning_rate: float,
            test: bool = False
    ):
        # allocate memory for training
        self._allocate_train_workspace(batch_sz)
        assert(self.ws is not None)

        # trim train data to fit an exact number of batches
        train_full_sz = data_source.tr_d[0].shape[0]  
        train_sz = train_full_sz - (train_full_sz % batch_sz)
        for epoch in range(epochs):
            print(f"Epoch {(epoch+1):3}: ", end="", flush=True)

            # shuffle data before creating the minibatches
            if epoch != 0:
                data_source.shuffle_training_data()

            # do one gradient step per mini_batch
            X,Y = data_source.tr_d
            for start_idx in range(0, train_sz, batch_sz):
                end_idx = start_idx+batch_sz

                # clear all bias and weight gradient accumulators
                for b_gr,w_gr in zip(self.ws.b_grad[1:], self.ws.w_grad[1:]):
                    b_gr.zero_()
                    w_gr.zero_()
                
                # Do forward and backward passes for all samples in the
                # mini_batch, accumulating the gradients produced by each
                # sample.
                self.ws.layers[0].copy_(X[start_idx:end_idx])
                self._feedforward()
                self._backprop(Y[start_idx:end_idx])
                self._update_parameters(batch_sz, learning_rate)

            # test how good the network is doing
            if (test or epoch+1 == epochs) and data_source.te_d is not None:
                test_data = data_source.te_d
                correct, total = self._evaluate(test_data) 
                perc = 100 * correct / total
                print(f"{perc:6.2f}%")
                if abs(perc - 100) < 0.0000001:
                    print("good enough!")
                    break
            else:
                print(f"complete.")

        # Done training. Allocate space to predict only
        self._allocate_prediction_workspace()
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

    def describe(self, layers=False):
        s = []
        w_parts = ["\nWEIGHTS:"]
        for i, w in enumerate(self.weights):
            ob = w if layers else ''
            w_parts.append(f'  L{i} {list(w.shape)}\n  {ob}')
        w_parts = "\n".join(w_parts)
        s.append(w_parts)
        
        b_parts = ["\nBIASES:"]
        for i,b in enumerate(self.biases):
            ob = b if layers else ''
            b_parts.append(f'  L{i} {list(b.shape)}\n  {ob}')
        b_parts = "\n".join(b_parts)
        s.append(b_parts)


        if self.ws is None:
            s.append("\nNO WORKSPACE")
            return "\n".join(s)

        batch_sz = self.ws.batch_sz
        ws_batch_sz = [f"\nWORKSPACE: BATCH_SZ={batch_sz}"]
        ws_batch_sz = "\n".join(ws_batch_sz)
        s.append(ws_batch_sz)
        
        ws_activs_parts = ["\nWS.ACTIVS:"]
        for i, a in enumerate(self.ws.layers):
            ob = a if layers else ''
            ws_activs_parts.append(f'  L{i} {list(a.shape)}\n  {ob}')
        ws_activs_parts = "\n".join(ws_activs_parts)
        s.append(ws_activs_parts)
        
        ws_aprime_parts = ["\nWS.A_PRIME", f"  {list(self.ws.a_prime.shape)}"]
        ws_agrad_parts = ["\nWS.A_GRAD", f"  {list(self.ws.a_grad.shape)}"]
        s.extend(ws_aprime_parts)
        s.extend(ws_agrad_parts)
        
        ws_wgrad_parts = ["\nWS.W_GRADS:"]
        for i, wg in enumerate(self.ws.w_grad):
            ob = wg if layers else ''
            ws_wgrad_parts.append(f'  L{i} {list(wg.shape)}\n  {ob}')
        ws_wgrads_parts = "\n".join(ws_wgrad_parts)
        s.append(ws_wgrads_parts)
        
        ws_bgrad_parts = ["\nWS.B_GRADS:"]
        for i, bg in enumerate(self.ws.b_grad):
            ob = bg if layers else ''
            ws_bgrad_parts.append(f'  L{i} {list(bg.shape)}\n {ob}')
        ws_bgrads_parts = "\n".join(ws_bgrad_parts)
        s.append(ws_bgrads_parts)
        
        return "\n".join(s)
        
    def __str__(self):
        return self.describe(layers=False)
                
def profile_network(nn: NeuralNetwork,
                    data_source: MnistLoader,
                    epochs: int,
                    batch_size: int,
                    test: bool):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ],
        acc_events=True
    ) as prof:
        nn.stoc_gradient_descent(
            data_source=data_source,
            epochs=epochs,
            batch_sz=batch_size,
            learning_rate=0.5,
            test=test
        )

    events = prof.key_averages()
    print(events.table(sort_by="self_cpu_time_total", row_limit=20))
    prof.export_chrome_trace("trace.json")
    return

def main(data_path: str = "mnist.pkl.gz",
         params_path: str | None = "mnist_params",
         profile: bool = False):
    print("PROBING SYSTEM")
    print("XPU available?:", torch.xpu.is_available())

    
    print("\nLOADING DATA")
    mnist = MnistLoader(data_path)
    if profile:
        mnist.mini_sample()
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
        print("\n(PROFILING)")
        profile_network(
            nn,
            data_source=mnist,
            epochs=epochs,
            batch_size=batch_size,
            test=do_test
        )
        print("\nDONE. BYE")
        exit(0)

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
