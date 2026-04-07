import pickle
import gzip

import numpy as np

# Activation and activation prime functions
def sigmoid(z, out):
    # out = 1 / (1 + exp(-z))
    np.negative(z, out=out)
    np.exp(out, out=out)
    np.add(out, 1.0, out=out)
    np.reciprocal(out, out=out)
    return
def sigmoid_prime_from_sigm(a, z, out):
    # out = a * (1-a)
    np.subtract(1.0, a, out=out)
    np.multiply(a, out, out=out)
    return
def linear(z, out):
    # out = z
    out[:] = z
    return
def linear_prime(a, z, out):
    # out = 1
    out.fill(1.0)
    return

# Decision and decision inverse functions
def identity(a) -> int: 
    return a.ravel()
def argmax(a) -> int: 
    return int(np.argmax(a))
def argmax_inv(y) -> np.ndarray:
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

# Equality tests functions
def equal_float_ndarray(y_hat, y) -> bool:
    return np.allclose(y_hat, y, atol=1e-6)
def equal_scalar(y_hat, y) -> bool:
    return y_hat == y

# Cost and cost derivative functions
def mse(y_hat, y) -> float:
    # return = (1/2) * Sum((y_hat - y)^2)
    diff = y_hat - y
    return 0.5 * np.sum(diff * diff)
def mse_prime(y_hat, y, out):
    # out = y_hat - y
    np.subtract(y_hat, y, out=out)
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
    "equal": equal_scalar
}

class MnistLoader(object):
    def __init__(self, path):
        try:
            f = gzip.open(path, 'rb')
        except FileNotFoundError as e:
            print(f"Could not find '{path}'.")
            exit(1)
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.tr_d, self.va_d, self.te_d = u.load()
        f.close()
        self.flatten_data()
    def zeros_one_at(self, i):
        """Return a 10-dimensional zero vector with 1.0 at the i-th index"""
        e = np.zeros((10, 1))
        e[i] = 1.0
        return e
    def flatten_data(self):
        for dname in ("tr_d", "va_d", "te_d"):
            X,Y = getattr(self, dname)
            flat_X = [np.reshape(x, (-1, 1)) for x in X]
            flat_Y = Y
            setattr(self, dname, list(zip(flat_X, flat_Y)))
        return
    def micro_sample(self, total, x_sz):
        start_idx = (784//2) - (x_sz//2)
        for dname in ("tr_d", "va_d", "te_d"):
            data_list = getattr(self, dname)
            small_data_list = [(x[start_idx:start_idx+x_sz],y)
                               for x,y in data_list[:total]]
            setattr(self, dname, small_data_list)
        return
    def describe(self):
        names = ("Training", "Validation", "Test")
        parts = (self.tr_d, self.va_d, self.te_d)
        for name, part in zip(names, parts):
            x,y = part[0]
            t = type(y)
            y_type = f"{t.__module__}.{t.__qualname__}"
            print(f"{name:10} ({len(part)} samples):\n"
                  f"    X:{x.shape} --> y:{y_type}")
        return

class NeuralNetwork(object):
    def __init__(self, weights, biases, layers, ly_sizes, funcs, dtype, rng):
        self.ly_sizes = ly_sizes
        self.rng = rng
        self.dtype = dtype
        self.weights = weights
        self.weights_acc_grad = [np.zeros(w.shape, dtype=self.dtype)
                                 for w in weights]
        self.biases = biases
        self.biases_acc_grad = [np.zeros(b.shape, dtype=self.dtype)
                                for b in biases]
        # activations during feedforwand. delta_z during backprop
        self.layers = layers

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
            
        # Buffers to keep temporary values
        max_ly_sz = max(ly_sizes)
        max_W_sz = max(l*ll for l, ll in
                       zip(self.ly_sizes[:-1], self.ly_sizes[1:]))
        # to hold the activation prime
        self.tmp_a_prime = np.zeros(max_ly_sz, dtype=self.dtype)
        # to hold the activation gradient
        self.tmp_a_grad  = np.zeros(max_ly_sz, dtype=self.dtype)
        # to hold the delta of a layer's weights
        self.tmp_w_grad = np.zeros(max_W_sz, dtype=self.dtype)
        return
    
    @classmethod
    def from_weights(cls, init_weights:list, init_biases:list, funcs,
                     dtype=np.float32, rand_seed=42):
        rng = np.random.default_rng(rand_seed)
        
        # create list of layer sizes
        ly_sizes = [len(init_weights[0][0])]
        ly_sizes.extend([len(W) for W in init_weights])

        # create layers
        layers = [np.zeros((l_sz,1), dtype=dtype) for l_sz in ly_sizes]
        
        # create weights
        weights = [np.array([], dtype=dtype)] # dummy for layer 0
        weights.extend([np.asarray(w, dtype=dtype) for w in init_weights])

        # create biaseses
        biases = [np.array([], dtype=dtype)] # dummy for layer 0
        biases.extend([np.asarray(b, dtype=dtype) for b in init_biases])

        # check sizes
        for i,(w,b) in enumerate(zip(weights, biases)):
            if w.shape[0] != b.shape[0]:
                raise ValueError(f"weights[{i}]={str(w.shape)} does not "
                                 f"match bias[{i}]={str(b.shape)}.")

        return cls(weights, biases, layers, ly_sizes, funcs, dtype, rng)
    
    @classmethod
    def from_layers(cls, ly_sizes:list, funcs, dtype=np.float32, rand_seed=42):
        # seed rng
        rng = np.random.default_rng(rand_seed)
        
        # create layers
        layers = [np.zeros((l_sz,1), dtype=dtype) for l_sz in ly_sizes]

        # weights and biases
        weights = [np.array([], dtype=dtype)] # dummy for layer 0
        biases  = [np.array([], dtype=dtype)] # dummy for layer 0
        for l_sz, ll_sz in zip(ly_sizes[:-1], ly_sizes[1:]):
            # extra column for biases
            xavier_max = 1/np.sqrt(l_sz)
            xavier_min = -xavier_max
            w = rng.uniform(xavier_min, xavier_max,
                            size=(ll_sz,l_sz)).astype(dtype)
            weights.append(w)
            b = rng.uniform(xavier_min, xavier_max,
                            size=(ll_sz,1)).astype(dtype)
            biases.append(b)
        return cls(weights, biases, layers, ly_sizes, funcs, dtype, rng)
    
    # These 'fn_' functions are to be replaced in __init__ with the
    # ones provided by the user
    def fn_hl_activ(self, z, out):
        # activation function of hidden layers
        return
    def fn_hl_activ_prime(self, a, z, out):
        # derivative of activation function of hidden layers
        return
    def fn_ol_activ(self, z, out):
        # activation function of output layer
        return
    def fn_ol_activ_prime(self, a, z, out):
        # derivative of activation function of output layer
        return
    def fn_decision(self, a) -> list:
        # interprets the last layer into the actual y_hat
        return []
    def fn_decision_inv(self, y):
        # expresses y as the expected values of the last layer
        return y
    def fn_equal(self, y_hat, y) -> int:
        # how to determine if they_hat is equal to y
        return 0
    def fn_loss(self, y_hat, y) -> float:
        return 0.0
    def fn_loss_prime(self, y_hat, y, out):
        # gradient of C with respect to output layer
        return
    def feedforward(self, x:list):
        # INPUT LAYER: simply assign the input data to this layer
        self.layers[0][:] = x

        # HIDDEN LAYERS: z_l = W_l times a_{l-1}; a_l = activ(z_l)
        for l in range(1, len(self.ly_sizes)-1):
            np.matmul(self.weights[l], self.layers[l-1], out=self.layers[l])
            np.add(self.layers[l], self.biases[l], out=self.layers[l])
            self.fn_hl_activ(self.layers[l], self.layers[l])

        # OUTPUT LAYER: same as hidden layers, but it may have a different
        # activation function
        np.matmul(self.weights[-1], self.layers[-2], out=self.layers[-1])
        np.add(self.layers[-1], self.biases[-1], out=self.layers[-1])
        self.fn_ol_activ(self.layers[-1], self.layers[-1])
        return
    def backprop(self, y):
        # Use the same space that now have the activations to store
        # the deltas.

        # STARTUP:
        # - compute delta_z of output layer
        out_layer = self.layers[-1]
        ol_sz = out_layer.shape[0]
        y_as_layer = self.fn_decision_inv(y)
        ol_a_prime = self.tmp_a_prime[:ol_sz].reshape(-1,1)
        ol_a_grad  = self.tmp_a_grad[:ol_sz].reshape(-1,1)
        self.fn_ol_activ_prime(a=out_layer, z=None, out=ol_a_prime)
        self.fn_loss_prime(y_hat=out_layer, y=y_as_layer, out=ol_a_grad)
        np.multiply(ol_a_prime, ol_a_grad, out=out_layer)

        # OUTPUT AND ALL BUT FIRST HIDDEN LAYERS:
        # - compute gradient of C w.r.t. this layer (tl) weights and biases.
        # - compute delta_z of previous layer (pl) (only hidden layers).
        lys = len(self.layers)
        for pl,tl in zip(range(lys-2, -1, -1), range(lys-1, 0, -1)):
            prev_layer = self.layers[pl]
            pl_sz = prev_layer.shape[0]
            this_layer = self.layers[tl]
            tl_sz = this_layer.shape[0]
            
            # gradient of C w.r.t. this layer biases
            tl_delta_z = self.layers[tl]
            acc_b_grad = self.biases_acc_grad[tl]
            np.add(acc_b_grad, tl_delta_z, out=acc_b_grad)
            
            # gradient of C w.r.t. this layer weights
            tl_w_grad = self.tmp_w_grad[:tl_sz*pl_sz].reshape(tl_sz,pl_sz)
            acc_w_grad = self.weights_acc_grad[tl]
            np.matmul(tl_delta_z, prev_layer.T, out=tl_w_grad)
            np.add(acc_w_grad, tl_w_grad, out=acc_w_grad)

            if pl > 0: # delta_z of previous *hidden* layers only
                pl_a_prime = self.tmp_a_prime[:pl_sz].reshape(-1,1)
                pl_a_grad = self.tmp_a_grad[:pl_sz].reshape(-1,1)
                tl_w = self.weights[tl]
                self.fn_hl_activ_prime(a=prev_layer, z=None, out=pl_a_prime)
                np.matmul(tl_w.T, tl_delta_z, out=pl_a_grad)
                np.multiply(pl_a_prime, pl_a_grad, out=prev_layer)
        return
    def update_parameters(self, mini_batch_sz: int, learning_rate):
        learn_factor = - learning_rate / mini_batch_sz
        # self.{weights,biases,..._acc_grad}[0] are dummy.
        for ly in range(1, len(self.weights)):
            b = self.biases[ly]
            b_gr = self.biases_acc_grad[ly]
            np.multiply(b_gr, learn_factor, out=b_gr)
            np.add(b, b_gr, out=b)

            w = self.weights[ly]
            w_gr = self.weights_acc_grad[ly]
            np.multiply(w_gr, learn_factor, out=w_gr)
            np.add(w, w_gr, out=w)
        return
    def evaluate(self, test_data) -> int:
        correct_count = 0
        for x,y in test_data:
            self.feedforward(x)
            out_layer = self.layers[-1]
            y_hat = self.fn_decision(out_layer)
            equal = int(self.fn_equal(y_hat, y)) 
            correct_count += equal
        return correct_count
    def stoc_gradient_descent(self, train_data, epochs, mini_batch_size,
                              learning_rate, test_data=None):
        train_sz = len(train_data)
        for epoch in range(epochs):
            print(f"Epoch {(epoch+1):3}: ", end="", flush=True)
            # randomize samples order
            self.rng.shuffle(train_data)
            # do one descent step for each mini-batch
            for start_idx in range(0, train_sz, mini_batch_size):
                end_idx = min(start_idx+mini_batch_size, train_sz)
                # clear all bias and weight gradiant acumulators
                for b_gr,w_gr in zip(self.biases_acc_grad[1:],
                                     self.weights_acc_grad[1:]):
                    b_gr.fill(0.0)
                    w_gr.fill(0.0)
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
                perc = 100 * self.evaluate(test_data) / len(test_data)
                print(f"{perc:6.2f}%")
                if abs(perc - 100) < 0.0000001:
                    print("good enough!")
                    break
            else:
                print(f"complete.")
        return
    def save_params(self, path:str):
        content = {}
        for i in range(1, len(self.weights)): # skip dummy index 0
            content[f"W{i}"] = self.weights[i]
            content[f"b{i}"] = self.biases[i]
        np.savez_compressed(path, **content)
        return
    @classmethod
    def load_params(cls, path:str):
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        # infer how many layers were saved
        weights = []
        biases  = []
        i = 1
        while f"W{i}" in data:
            weights.append(data[f"W{i}"])
            biases.append(data[f"b{i}"])
            i += 1
        return weights, biases
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

def train_from_layers(layers, data, epochs, minibatch_sz, lrate,
                      file_path: str | None = "saved_params"):
    
    nn = NeuralNetwork.from_layers(layers, funcs)
    nn.stoc_gradient_descent(data.tr_d, epochs, minibatch_sz,
                             lrate, test_data=data.te_d)
    if file_path:
        nn.save_params(file_path)
    return

def train_from_weights(weights, biases, data, epochs, minibatch_sz, lrate,
                      file_path="saved_params"):

    nn = NeuralNetwork.from_weights(weights, biases, funcs)
    nn.stoc_gradient_descent(data.tr_d, epochs, minibatch_sz,
                             lrate, test_data=data.te_d)
    if file_path:
        nn.save_params(file_path)
    return

def main(data_path="mnist.pkl.gz", params_path="mnist_params"):
    mnist = MnistLoader(data_path)
    
    print("TRAINING NETWORK FROM LAYER'S DESCRIPTION")
    in_ly_sz = [len(mnist.tr_d[0][0])]
    out_ly_sz = [10]
    all_layers = in_ly_sz + [200, 200] + out_ly_sz
    print("in", all_layers, "out")
    
    train_from_layers(layers=all_layers, data=mnist, epochs=100,
                      minibatch_sz=20, lrate=0.5, file_path=None)
    
    # print("TRAINING SECOND NETWORK FROM PREVIOUS WEIGHTS AND BIASES")
    # W,b = NeuralNetwork.load_params(params_path) 
    # train_from_weights(W, b, data=mnist, epochs=1, minibatch_sz=20,
    #                    lrate=0.5, file_path=params_path+"_new")

    print("DONE!")

if __name__ == "__main__":
    main(params_path=None)
