from __future__ import annotations
import pathlib, os, torch, pickle, copy, json, shutil, textwrap
from typing import cast, Mapping, Any, override
from datetime import datetime
from torch import Tensor, nn

torch.set_printoptions(precision=3, sci_mode=False, linewidth=512)
FLOAT_DTYPE = torch.float32
_TIMESTAMP = "TIMESTAMP"
# percentage drop considered a network collapse, triggering exit
_COLLAPSE_THRESHOLD = 50
_ALL_PLOTS_DIR_PATH = "all_plots"
_LABEL_CH_WIDTH = 100

# ------------------------------------------------------------------------
# Data Loaders. Keeps whole dataset in memory.
# ------------------------------------------------------------------------
class Loader(object):
    """Base class for dataset loaders.

    Subclasses must populate:
    - data_train, data_valid, data_test : tuple[Tensor, Tensor]
        Each is (X, Y) where X is float32 shaped (-1, C, H, W) and
        Y is int32 shaped (-1, 1).
    - img_dims : tuple[int, ...]
        Image shape as (channels, height, width). Used for reshaping flat
        vectors back to spatial tensors during augmentation and export.
    - in_ly_sz : int
        Number of input features (flattened image size).
    - out_ly_sz : int
        Number of output classes.
    - data_train_alt_X/Y : Tensor
        Pre-allocated buffers for in-place shuffle. Same shape as
        data_train tensors. augment_training_data re-allocates these tensors.
    """
    def __init__(
            self,
            data_params: dict[str, Any],
            path: str | None = None,
            dtype: torch.dtype = FLOAT_DTYPE
    ):
        self.dtype: torch.dtype
        self.img_dims: tuple
        self.in_ly_sz: int
        self.out_ly_sz: int
        self.data_train: tuple[Tensor, Tensor]
        self.data_valid: tuple[Tensor, Tensor]
        self.data_test: tuple[Tensor, Tensor]
        self.data_train_alt_X: Tensor
        self.data_train_alt_Y: Tensor
        self.labels: list[str]
        return
    
    def _export_img(
            self,
            export: int,
            exp_dir: str,
            data_augmented: tuple[list[Tensor], list[Tensor]] | None = None,
    ):
        """Exports 'export' samples to directory 'exp_dir'. If 'data_augmented'
        is not None, then exports side-by-side with their originals
        in self.data_train.
        """
        import matplotlib.pyplot as plt

        # remove old exported images
        os.makedirs(exp_dir, exist_ok=True)
        for p in pathlib.Path(exp_dir).iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        
        dpi = 24
        channels = self.img_dims[0]
        if channels == 1:
            cmap_val = "gray"
        elif channels == 3:
            cmap_val = None # ignored for RGB
        else:
            raise ValueError("Unexpected number of channels "
                            f"(self.dims[0]={channels})")
        
        # just export training data
        if data_augmented is None:
            tr_X,tr_Y = self.data_train
            N = min(export, tr_X.shape[0])
            for i in range(N):
                y = int(tr_Y[i].item())
                if channels == 1:
                    img0 = tr_X[i, 0].cpu().numpy()
                else:
                    img0 = tr_X[i].permute(1,2,0).cpu().numpy()

                # export each image
                fig, ax0 = plt.subplots(1, 1, figsize=(1, 1))
                ax0.imshow(img0, cmap=cmap_val); ax0.axis('off')
                label_suffix = ""
                if hasattr(self, "labels"):
                    label_suffix = f"_{self.labels[y]}"
                filename = f"{exp_dir}/{i:04}_{y}{label_suffix}.png"
                plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99,
                                    top=0.99, bottom=0.01)
                fig.savefig(filename, dpi=self.img_dims[-1]*dpi)
                plt.close(fig)
                        
        # export training and augmented data side-by-side
        else:
            tr_X,tr_Y = self.data_train
            au_X = torch.cat(data_augmented[0])
            N = min(export, tr_X.shape[0], au_X.shape[0])
            for i in range(N):
                y = int(tr_Y[i].item())
                if channels == 1:
                    img0 = tr_X[i, 0].cpu().numpy()
                    img1 = au_X[i, 0].cpu().numpy()
                else:
                    img0 = tr_X[i].permute(1,2,0).cpu().numpy()
                    img1 = au_X[i].permute(1,2,0).cpu().numpy()

                # export each image
                fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(2, 1))
                ax0.imshow(img0, cmap=cmap_val); ax0.axis('off')
                ax1.imshow(img1, cmap=cmap_val); ax1.axis('off')
                label_suffix = ""
                if hasattr(self, "labels"):
                    label_suffix = f"_{self.labels[y]}"
                filename = f"{exp_dir}/{i:04}_{y}{label_suffix}.png"
                plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99,
                                    top=0.99, bottom=0.01)
                fig.savefig(filename, dpi=self.img_dims[-1]*dpi)
                plt.close(fig)
        print(f"{export} samples exported to {exp_dir}.\nBye!")
        exit(0)

    def _build_transform(self, augm_params: dict[str, Any]):
        import kornia.augmentation as K
        tr_seq = []
        avail_transf = {
            "elastic": K.RandomElasticTransform, 
            "crop"   : K.RandomCrop,
            "hflip"  : K.RandomHorizontalFlip,
            "affine" : K.RandomAffine,
            "color"  : K.ColorJitter,
        }
        # follow the order given by the user
        augm_params = {} if augm_params is None else augm_params
        for given_key,given_params in augm_params.items():
            if given_key in avail_transf:
                transf = avail_transf[given_key]
                if not given_params: # e.g. hflip, has not parameters
                    tr_seq.append(transf())
                else:
                    tr_seq.append(transf(**given_params))

        # if the user gave no matching transformation keys
        if len(tr_seq) == 0:
            if len(augm_params) > 0:
                print("Note: None of the given transformation keys matched!\n"
                      f"keys: {list(augm_params.keys())}")
            return torch.nn.Identity()
        
        return K.AugmentationSequential(*tr_seq, same_on_batch=False)
        

    def augment_training_data(self, data_params: dict[str, Any]):
        """Expands data_train by a number `factor`, and shuffle the result.
        data_params keys:
        - factor      : augment data_train by this factor.
                        - 0.3 : trim the data to 30% of its original size. 
                        - 1   : do nothing.
                        - 2   : duplicate the original size.
                        - 2.7 : 2.7x the original data.
        - export      : Number of samples to export. if 0, then do not export.
                        Exporting exits the program. If factor > 1, then
                        exports the original and augmented side by side.
        - exp_dir     : Directory to export samples to.
        - augm_params : parameters for the elastic and affine transformations.

        At the end, it adds the key "train_sz" to the dict, with the updated
        size of the training dataset.
        """
        factor = data_params["factor"]
        export = int(data_params["export"])
        exp_dir = data_params["exp_dir"]
        augm_params = data_params["augm_params"]

        if factor <= 0:
            raise ValueError("data_params['factor'] must be positive")
        if export < 0:
            raise ValueError("data_params['export'] must be int zero or positive")
        
        # Trim dataset
        if factor <= 1:
            print(f"Trimming Dataset: {factor}x")
            if factor < 1:
                new_train_sz = int(self.data_train[0].shape[0] * factor)
                self.data_train = (self.data_train[0][:new_train_sz],
                                   self.data_train[1][:new_train_sz])
            if 0 < export:
                self._export_img(export, exp_dir)
    
        # Augment dataset
        if 1 < factor:
            print(f"Augmenting Dataset: {factor}x", end="", flush=True)
            transform = self._build_transform(augm_params)
            tr_X, tr_Y = self.data_train
            
            factor_int = int(factor)
            factor_dec = factor - factor_int
            N = tr_X.shape[0]
            aug_X, aug_Y = [], []
            try:
                for _ in range(factor_int-1):
                    print(".", end="", flush=True)
                    aug_X.append(transform(tr_X))
                    aug_Y.append(tr_Y)
                tail = int(factor_dec*N)
                if tail:
                    aug_X.append(transform(tr_X[:tail]))
                    aug_Y.append(tr_Y[:tail])
                print()
            except KeyboardInterrupt:
                print("\nInterrupted!")
                exit(1)
            if export > 0:
                self._export_img(export, exp_dir, data_augmented=(aug_X, aug_Y))
                
            full_X = torch.cat([tr_X]+aug_X, dim=0)
            full_Y = torch.cat([tr_Y]+aug_Y, dim=0)
            idx = torch.randperm(full_X.shape[0], device=tr_X.device)
            self.data_train = (full_X[idx], full_Y[idx])

        data_params["train_sz"] = self.data_train[0].shape[0]
        self.data_train_alt_X = torch.empty_like(self.data_train[0])
        self.data_train_alt_Y = torch.empty_like(self.data_train[1])
        
        return

    def shuffle_training_data(self):
        """Shuffle training data in-place using pre-allocated alt buffers.
    
        Avoids allocation by writing into data_train_alt_X/Y, then swapping
        the tuple references so data_train points to the shuffled copy.
        """
        X, Y = self.data_train
        idx = torch.randperm(X.shape[0], device=X.device)
        torch.index_select(X, dim=0, index=idx, out=self.data_train_alt_X)
        torch.index_select(Y, dim=0, index=idx, out=self.data_train_alt_Y)
        tmp = self.data_train
        self.data_train = (self.data_train_alt_X, self.data_train_alt_Y)
        self.data_train_alt_X, self.data_train_alt_Y = tmp
        return

    def describe(self):
        """Shows the shape and dtype of the training, validation and test
        datasets."""
        names = ("Training", "Validation", "Test")
        parts = (self.data_train, self.data_valid, self.data_test)
        for name, (x, y) in zip(names, parts):
            print(f"{name}:(X,Y):\n"
                  f"    X   {str(x.dtype):>14}   {list(x.shape)}\n"
                  f"    Y   {str(y.dtype):>14}   {list(y.shape)}")
        return

class MnistLoader(Loader):
    @override
    def __init__(
            self,
            data_params: dict[str, Any],
            path: str | None = None,
            dtype: torch.dtype = FLOAT_DTYPE
    ):
        import gzip
        self.dtype = dtype
        self.img_dims = (1, 28, 28)
        self.in_ly_sz = self.img_dims[0] * self.img_dims[1]\
            * self.img_dims[2]
        self.out_ly_sz = 10 # ten classes

        # read file
        path = path or "mnist.pkl.gz"
        try:
            with gzip.open(path, 'rb') as fo:
                tr, va, te = pickle.load(fo, encoding='latin1')
        except FileNotFoundError as e:
            print(f"Could not find '{path}'.")
            exit(1)
            
        device = "cpu"
        self.data_train = (
            torch.tensor(tr[0], dtype=self.dtype, device=device)\
            .view(-1, *self.img_dims),
            torch.tensor(tr[1], dtype=torch.int32, device=device).view(-1,1))
        self.data_valid = (
            torch.tensor(va[0], dtype=self.dtype, device=device)\
            .view(-1, *self.img_dims),
            torch.tensor(va[1], dtype=torch.int32, device=device).reshape(-1,1))
        self.data_test = (
            torch.tensor(te[0], dtype=self.dtype, device=device)\
            .view(-1, *self.img_dims),
            torch.tensor(te[1], dtype=torch.int32, device=device).reshape(-1,1))

        # update dataset size and allocate space for alt buffers
        data_params["train_sz"] = self.data_train[0].shape[0]
        self.data_train_alt_X = torch.empty_like(self.data_train[0])
        self.data_train_alt_Y = torch.empty_like(self.data_train[1])
        return

class CifarLoader(Loader):
    @override
    def __init__(
            self,
            data_params: dict[str, Any],
            path: str | None = None,
            dtype: torch.dtype = FLOAT_DTYPE
    ):
        import numpy as np
        import tarfile
        self.dtype = dtype
        self.img_dims = (3, 32, 32)
        self.in_ly_sz = self.img_dims[0] * self.img_dims[1]\
            * self.img_dims[2]
        self.out_ly_sz = 10 # ten classes

        # extract tar.gz
        tarpath = path or "cifar-10-python.tar.gz"
        extract_path = "cifar-10"
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        try:
            with tarfile.open(tarpath, "r:gz") as tar:
                tar.extractall(path=extract_path)
        except FileNotFoundError as e:
            print(f"Could not find '{tarpath}'.")
            exit(1)
        path = os.path.join(extract_path, "cifar-10-batches-py")

        # import each data batch
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        X_list, Y_list = [], []
        for i in (1, 2, 3, 4, 5):
            data_train_batch = unpickle(f"{path}/data_batch_{i}")
            X_list.append(data_train_batch[b'data'])
            Y_list.extend(data_train_batch[b'labels'])
        X_list = np.concatenate(X_list)
        
        # Divide to map values from (0, 255) to (0.0, 1.0)
        device = "cpu"
        X_all = torch.tensor(X_list, dtype=self.dtype, device=device)\
                     .view(-1, *self.img_dims)/255.0
        Y_all = torch.tensor(Y_list, dtype=torch.int32, device=device)\
                     .reshape(-1, 1)

        # set aside last training samples as validation. At most, half of the
        # training dataset
        valid_sz = min(int(0.5 * X_all.shape[0]), data_params["valid_sz"])
        train_sz = X_all.shape[0] - valid_sz 
        self.data_train = (X_all[:train_sz],  Y_all[:train_sz])
        self.data_valid = (X_all[train_sz:],  Y_all[train_sz:])

        # test set
        test = unpickle(f"{path}/test_batch")
        # Divide to map values from (0, 255) to (0.0, 1.0)
        X_te = torch.tensor(test[b'data'], dtype=self.dtype, device=device)\
                    .view(-1, *self.img_dims)/255.0
        Y_te = torch.tensor(test[b'labels'], dtype=torch.int32, device=device)\
                    .reshape(-1,1)
        self.data_test = (X_te, Y_te)

        # update dataset size, and allocate space for alt buffers
        data_params["train_sz"] = self.data_train[0].shape[0]
        self.data_train_alt_X = torch.empty_like(self.data_train[0])
        self.data_train_alt_Y = torch.empty_like(self.data_train[1])

        # label names
        meta = unpickle(f"{path}/batches.meta")
        self.labels = [n.decode() for n in meta[b'label_names']]
        return


# ------------------------------------------------------------------------
# Base module. User needs to implement forward and fit functions. 
# ------------------------------------------------------------------------
class Module(nn.Module):
    """Base class for neural network models.
    
    Subclasses must override:
        __init__(hyp_params, model_fns, data_source)
        forward(input_batch) -> Tensor
        fit(data_source) -> None
    
    Provides: evaluate, monitor_accuracy, and describe.

    Specific hyper-parameters to be used by the fit function are passed
    though the 'hyp_params' dictionary.
    """
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
            model_fns: Mapping[str, Any],
            data_source: Loader,
    ):
        super().__init__()

        # set activation, loss and decision functions
        self.fn_activ_hl = model_fns["activ_hl"]
        self.fn_loss = model_fns["loss"]
        self.fn_decision = model_fns["decision"]

        # save complete hyperparameters dictionary
        self.hyp_params = hyp_params
        self.device = hyp_params["device"]
        
        # best set of parameters and accuracy log
        self.best_state_dict: dict[str, Tensor] | None = None
        self.best_valid_acc: float = 0.0
        self.last_valid_acc: float = 0.0
        self.accuracy_log: list[tuple[float,float]] = []
        return

    def forward(self, input_batch: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def fit(self, data_source: Loader) -> None:
        raise NotImplementedError("Subclasses must implement fit()")

    def evaluate(
            self,
            dataset: tuple[Tensor, Tensor],
            batch_sz: int | None = None
    ):
        """Predicts the output for each sample in dataset[0], and compare
        it with the correct output in dataset[1]. Returns a tuple with the count
        of correct and total samples.

        If len(dataset[0]) is not a multiple of batch_sz, the dataset is trimmed. This
        is the 'total samples' value returned.
        """
        was_training = self.training
        self.eval()
        # default batch size
        if batch_sz is None:
            batch_sz = dataset[0].shape[0]

        # trim dataset to an exact number of batches
        test_full_sz = dataset[0].shape[0]
        test_sz = test_full_sz - (test_full_sz % batch_sz)

        # evaluate in batches
        with torch.no_grad():
            X,Y = dataset
            correct_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
            for start_idx in range(0, test_sz, batch_sz):
                end_idx = start_idx+batch_sz
                X_batch = X[start_idx:end_idx].to(self.device)
                Y_batch = Y[start_idx:end_idx].to(self.device)
                out_layer = self(X_batch)
                y_hat_batch = self.fn_decision(out_layer)
                correct_count += torch.sum(y_hat_batch == Y_batch)
        if was_training:
            self.train()
        return (int(correct_count.item()), test_sz)

    def monitor_accuracy(self,
                            train_samples: tuple[Tensor, Tensor],
                            valid_samples: tuple[Tensor, Tensor],
                            batch_sz: int,
                            save_best: bool = False,
                         ) -> tuple[str, bool]:
        """Evaluate and log train and validation accuracy.

        Checkpoints parameters if best performing seen so far (if save_best
        is True). Triggers early stop if validation accuracy reaches ~100%
        or if the model collapsed (drop exceeding _COLLAPSE_THRESHOLD from
        last call).

        Returns a message string with train/validation accuracy and a bool
        indicating whether training should stop early.
        """
        # evaluate accuracy for training/validation samples
        train_correct, train_total = self.evaluate(train_samples, batch_sz)
        valid_correct, valid_total = self.evaluate(valid_samples, batch_sz)
        train_acc = 100 * train_correct / train_total
        valid_acc = 100 * valid_correct / valid_total
        self.accuracy_log.append((train_acc, valid_acc))

        # generate message to show
        mark = ""
        if valid_acc > self.best_valid_acc:
            mark = " <-- new best!"
            self.best_valid_acc = valid_acc
            if save_best:
                self.best_state_dict = copy.deepcopy(self.state_dict())
        # safe float equality as float comes from int / int
        elif valid_acc == self.best_valid_acc:
            mark = " <-- best"
        msg = f"tra:{train_acc:6.2f}%   val:{valid_acc:6.2f}%"+mark

        # early exit: good enough or collapse
        do_break = False
        if abs(100 - valid_acc) < 0.0001:
            msg += "Validation is good enough! Early exit."
            do_break = True

        # check if network collapsed
        if self.last_valid_acc - valid_acc > _COLLAPSE_THRESHOLD:
            msg+=f"\nThe Network Collapsed (+{_COLLAPSE_THRESHOLD}% drop)!"
            do_break = True
        self.last_valid_acc = valid_acc
        return (msg, do_break)

    def describe(self):
        """Print human-readable parameters given to the model."""
        to_print = []
        for n,p in self.hyp_params.items():
            # print only human-understandable parameters
            if isinstance(p, (int, float, tuple, list)):  
                to_print.append((n,p))
        name_width = max(len(n) for n,_ in to_print)
        for n,p in to_print:
            print(f"  {n:{name_width}} : {str(p)}")
        return


# ------------------------------------------------------------------------
# Utility functions.
# ------------------------------------------------------------------------

def _build_plot_labels(filenames: list[str], keys: list[str]
                       ) -> tuple[list[str], str]:
    """Reads the metadata of each file in 'filenames', and build the list of labels
    for each plot, plus a string with all the parameters that are common across all
    plots (to be placed as subtitle).

    - filenames: paths to JSON plot data files. the labels will be returned in the
                 same order.
    - keys     : keys to sort parameters within each label. Keys not in this list
                 are appended at the end of the label.

    Returns:
    - labels: one legend label string per file.
    - common: string with all the common key:value parameters.
    """
    # collect list of metadatas across files. add accuracy to it
    accuracy_test_key = "test_accu"
    meta_per_file: list[dict[str,Any]] = []
    for f in filenames:
        meta, accu = _load_plot_data(f)
        meta[accuracy_test_key] = accu["test"]
        meta_per_file.append(meta)
    
    # collect all keys across all files
    all_keys_found: set[str] = set()
    excluded_keys = {"title", "about"}
    for meta in meta_per_file:
        for k in meta:
            if k not in all_keys_found and\
               k not in excluded_keys:
                all_keys_found.add(k)
        

    # sort according to given keys list, and leave the rest at the end
    # put accuracy test at the beginning.
    all_keys_found_sorted = []
    for k in [accuracy_test_key] + keys:
        if k in all_keys_found:
            all_keys_found_sorted.append(k)
            all_keys_found.remove(k)
    all_keys_found_sorted.extend(list(sorted(all_keys_found)))

    # extract common keys
    keys_common: list[str] = [] # in all dicts with same value.
    keys_diff_val: list[str] = [] # in all dicts, but diff value.
    keys_unique: list[str] = [] # not in all dicts.
    vals_width: dict[str,int] = {} # max value-width for each key.
    for k in all_keys_found_sorted:
        is_common = True
        is_unique = False
        common_val = None
        v_wd_max = 0
        for meta in meta_per_file:
            if k not in meta:
                is_unique = True
                is_common = False
                continue
            v = meta[k]
            
            # determine value width
            if k == "test_accu":
                v_wd = 5
                is_common = False # always goes to legend
            else:
                if isinstance(v, (list,tuple)):
                    v = str(v).replace(" ","")
                    meta[k] = v
                v_wd = len(str(meta[k]))
            v_wd_max = max(v_wd_max, v_wd)

            # determine if value is common
            if common_val is None:
                common_val = v
            if common_val != v:
                is_common = False
                
        vals_width[k] = v_wd_max
        if is_common:
            keys_common.append(k)
        elif is_unique:
            keys_unique.append(k)
        else:
            keys_diff_val.append(k)

    # Create the labels with the diff and unique key:value pairs.
    labels = []
    keys_unique.sort()
    error_keys = ["EXCEPTION", "TRUNCATED"]
    for meta in meta_per_file:
        parts = []

        # first the shared keys with different value
        for k in keys_diff_val:
            if k in error_keys:
                parts.append(f"{k}")
                continue
            v, w = meta[k], vals_width[k]
            if k == "test_accu":
                v_str = f"{v:>0.{w-3}f}%"
            elif isinstance(v, float):
                v_str = f"{v:0.{w-2}f}"
            elif isinstance(v, int):
                v_str = f"{v:>{w}}"
            else:
                v_str = f"{v:<{w}}"
            parts.append(f"{k}:{v_str}")

        # then, unique unique keys
        for k in keys_unique:
            if k not in meta:
                continue
            if k in error_keys:
                parts.append(f"{k}")
                continue
            v, w = meta[k], vals_width[k]
            if k == "test_accu":
                v_str = f"{v:>0.{w-3}f}%"
            elif isinstance(v, (list,tuple)):
                v_str = str(v).replace(" ", "")
                v_str = f"{v_str:<{w}}"
            elif isinstance(v, float):
                v_str = f"{v:0.{w-2}f}"
            else:
                v_str = f"{v:>{w}}"
            parts.append(f"{k}:{v_str}")
            
        labels.append(" │ ".join(parts))

    # put together all the common parameters
    common = " │ ".join(f"{k}:{meta_per_file[0][k]}" for k in keys_common)

    return labels, common

def _save_plot_data(path: str,
                   metadata: Mapping[str, Any],
                   accuracy: Mapping[str, Any],
                   ) -> None:
    """Save json file with the model's hyperparameters and its train
    and validation accuracy at each epoch; plus its final test accuracy.
    
    - metadata: dictionary with the hyper-parameters to save.
    - accuracy: dictionary with train and validation accuracy series,
                plus test accuracy scalar.

    This file can then be loaded with '_load_plot_data()' and plotted
    with function 'plot_all()'
    """
    path_stub, path_ext = os.path.splitext(path)
    for k in ["EXCEPTION", "TRUNCATED"]:
        if k in metadata:
            path_stub += f"_{k}"
    
    payload = {
        "metadata" : metadata,
        "accuracy" : accuracy
    }
    
    path = path_stub + path_ext
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return

def _load_plot_data(
        path: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load plot data saved with '_save_plot_data()'"""
    with open(path, 'r') as f:
        payload = json.load(f)
    return (payload["metadata"], payload["accuracy"])

def _copy_to_dir(src_path: str, dest_dir: str):
    """Make a copy of 'src_path' file into dest_dir, making sure its
    destination name is unique (based on the 'src_path')""" 
    os.makedirs(dest_dir, exist_ok=True)
    unique_name = "_".join(pathlib.Path(src_path).parts)
    shutil.copy2(src_path, os.path.join(dest_dir,unique_name))
    return

def random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across CPU, CUDA, and XPU.
    The default seed being the meaning of the universe (42)."""
    torch.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

def model_train_eval(
        general: dict[str, Any],
        data_params: dict[str, Any],
        hyp_params: dict[str, Any],
        plot_params: dict[str, Any],
        do_save_params: bool = False,
        do_save_plot_data: bool = True,
):
    """Load data, train a model, evaluate on test set, and save results.
        - general    : experiment metadata shown in plot titles.
                       Required keys: "title", "about", "prefix" (output path prefix).
        - data_params: dataset configuration.
                       Required keys: "dataset" ("mnist" or "cifar-10"), "device",
                       "factor" (augmentation), "export", "exp_dir", "augm_params".
                       For datasets that only have train and test datasets, you can
                       add the key "valid_sz":N to use the last N training samples
                       as the validation dataset (default min(0.5*train_sz, 5000)).
                       When data is loaded, the key "train_sz" is added.
        - hyp_params : model and training hyperparameters.
                       The only required keys are:
                           "batch_sz", "model" (your Model subclass).
                       The default Model.__init__ uses:
                           "hid_layers", "model_fns"
                       If you are loading parameters from a file:
                           "load_path"
                       The rest are hyperparameters that you may want to use in your
                       Module.__init__(), .forward(), and .fit(). For example:
                            "epochs", "lrn_rate", "w_decay", "betas", "dropout".
                       Keys listed in plot_params["keys_hparam"] are saved to JSON.
        - plot_params: controls which parameters from the previous dictionaries are
                       saved to JSON (if 'do_save_plot_data' is set):
                            "keys_general", "keys_data", "keys_hparam"
        - do_save_params: if True, save best model weights to a .params file.
        - do_save_plot_data: if True, save metadata and accuracy log to JSON.
    """
    global _TIMESTAMP
    _TIMESTAMP = datetime.now().strftime("%m%d-%H%M%S")
    # PROBE SYSTEM
    device = hyp_params["device"]
    if device != "cpu":
        print("\nXPU AVAILABLE:", torch.xpu.is_available())

        
    # LOAD DATA
    dataset = data_params["dataset"]
    print(f"\nLOADING DATA: {dataset}")
    if dataset == "mnist":
        data_source = MnistLoader(data_params)
    elif dataset == "cifar-10":
        if "valid_sz" not in data_params:
            data_params["valid_sz"] = 5000
        data_source = CifarLoader(data_params)
    else:
        raise RuntimeError(f"Invalid dataset '{dataset}'")
    data_source.augment_training_data(data_params)
    data_source.describe()

    
    # CREATE MODEL
    NeuralNet = hyp_params["model"]
    model_fns = hyp_params["model_fns"]
    print("\nCREATING MODEL")
    model: Module = NeuralNet(hyp_params, model_fns, data_source)

    
    # LOAD PARAMETERS AND MOVE TO DEVICE
    load_params_path = hyp_params["load_path"]
    if isinstance(load_params_path, str):
        print(f"LOADING PARAMS: {load_params_path}")
        model.load_state_dict(torch.load(load_params_path, weights_only=True))
    if device != "cpu":
        model.to(device)

    
    # TRAIN
    try:
        print(f"\nTRAINING")
        model.describe()
        model.fit(data_source)
    except KeyboardInterrupt:
        print("\n\n---- INTERRUPTED! ----\n")
        general["TRUNCATED"] = True
    except Exception as e:
        print("\n\n---- EXCEPTION! ----\n")
        print(e)
        general["EXCEPTION"] = True


    # EVALUATE TEST DATASET
    batch_sz = hyp_params["batch_sz"]
    test_corr, test_tot = model.evaluate(data_source.data_test, batch_sz)
    test_accu = 100 * test_corr / test_tot

    
    # SAVE MODEL PARAMETERS
    prefix = general["prefix"]
    dirname = os.path.dirname(prefix)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    save_stub = f"{prefix}{test_accu:06.2f}__{_TIMESTAMP}"
    if do_save_params:
        save_params_path = save_stub+".params"
        print(f"\nSAVING PARAMS: {save_params_path}")
        torch.save(model.best_state_dict, save_params_path)


    # SAVE PLOT DATA
    if do_save_plot_data:
        save_plot_path = save_stub+".json"
        print(f"\nSAVING PLOT DATA: {save_plot_path}")
        metadata = {}
        for k in plot_params["keys_general"]:
            if k in general:
                metadata[k] = general[k]
        for k in plot_params["keys_data"]:
            if k in data_params:
                metadata[k] = data_params[k]
        for k in plot_params["keys_hparam"]:
            if k in hyp_params:
                metadata[k] = hyp_params[k]
        for k in ["EXCEPTION", "TRUNCATED"]:
            if k in general:
                metadata[k] = general[k]
        accuracy = {
            "train" : [tr for tr,_ in model.accuracy_log],
            "valid" : [va for _,va in model.accuracy_log],
            "test"  : test_accu,
        }
        _save_plot_data(save_plot_path,
                       metadata,
                       accuracy)
    return

def plot_all(
        plot_params: dict[str, Any],
        path_prefix: str,
) -> None:
    """Plot accuracy curves from all JSON files matching path_prefix*.json.

    The 'path_prefix' may contain directories which will be created. So, at
    the end, this function also copies the generated PNG to a common directory
    just for easy navigation of all plot generated across different calls.

    Each JSON file contributes:
        - one train accuracy curve.
        - one validation accuracy curve.
        - one horizontal test accuracy line.
    Parameters that vary across files appear in the legend; shared parameters
    appear in the subtitle.

    - plot_params: rendering configuration:
                     "size" (fig size), "x_range", "y_range" ("auto" or tuple),
                     "plot_train", "plot_valid", "plot_test" (bools),
                     "keys_general", "keys_data", "keys_hparam" (label ordering).
    - path_prefix: glob prefix for finding JSON files and naming the
                   output PNG. Output written to {path_prefix}plot.png.
    """
    
    import glob
    import matplotlib.pyplot as plt
    default_plot_params = {
        "keys_general" : [],
        "keys_data"    : [],
        "keys_hparam"  : [],
        "size"         : (12, 16),
        "x_range"      : "auto",
        "y_range"      : "auto",
        "plot_train"   : False,
        "plot_valid"   : True,
        "plot_test"    : True,
    }
    plot_params = default_plot_params | plot_params
    print(f"\nPLOTTING ALL '{path_prefix}*.json' FILES.")
    json_paths = sorted(glob.glob(f"{path_prefix}*.json"), key=os.path.getmtime)
    if not json_paths:
        print(f"No '{path_prefix}*.json' files found.")
        return

    keys = []
    keys.extend(plot_params["keys_general"])
    keys.extend(plot_params["keys_data"])
    keys.extend(plot_params["keys_hparam"])
    labels, common = _build_plot_labels(json_paths, keys)
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, ax = plt.subplots(figsize=plot_params["size"])
    for spine in ax.spines.values():
        spine.set_zorder(10)
    xrange = [float("inf"), float("-inf")]
    yrange = [float("inf"), float("-inf")]
    test_accurs = {}
    for i,(path,label) in enumerate(zip(json_paths, labels)):
        _, accuracy = _load_plot_data(path)
        train_accs = accuracy["train"]
        valid_accs = accuracy["valid"]
        test_accu  = accuracy["test"]
        epochs = [e+1 for e in range(len(train_accs))]
        color = plot_colors[i % len(plot_colors)]
        label = "\n".join(textwrap.wrap(label, width=_LABEL_CH_WIDTH))
        label = None if label == "" else label

        # plot lines!
        if plot_params["plot_train"]:
            ax.plot(epochs, train_accs,
                    linestyle="-",
                    linewidth=0.5,
                    color=color,
                    alpha=0.45,
                    zorder=1,
                    marker=None,)

        if plot_params["plot_valid"]:
            # find best validation value and mark it
            best_e_v = (0, 0)
            for e,v in enumerate(valid_accs):
                if best_e_v[1] <= v:
                    best_e_v = (e, v)
            best_epoch = epochs[best_e_v[0]]
            best_val_accu = best_e_v[1]
            alt_lab = f"val*:{best_val_accu:5.2f} | " + str(label) \
                if not plot_params["plot_test"] else None
            ax.plot(epochs, valid_accs,
                    linestyle="-",
                    linewidth=1.33,
                    color=color,
                    alpha=0.6,
                    zorder=2,
                    label=alt_lab,
                    marker=None,)
            ax.plot([best_epoch], [best_val_accu],
                    marker="o",
                    markersize=8,
                    color="white",
                    alpha=0.6,
                    zorder=3,
                    linestyle="None")
            ax.plot([best_epoch], [best_val_accu],
                    marker="*",
                    markersize=7,
                    color=color,
                    alpha=1.0,
                    zorder=3,
                    linestyle="None")
            
        if plot_params["plot_test"]:
            # shift in case it overlaps with another marker
            if test_accu in test_accurs:
                test_accurs[test_accu] -= 1
            else:
                test_accurs[test_accu] = 0
            accu_x = epochs[-1] - test_accurs[test_accu]
            ax.scatter(accu_x, test_accu,
                    marker='<',
                    s=150,
                    color="white",
                    alpha=0.6,
                    zorder=4,)
            ax.scatter(accu_x, test_accu,
                    marker='<',
                    s=100,
                    color=color,
                    alpha=1.0,
                    zorder=4,
                    label=label)

        #collect plot axes boundaries
        xrange[0] = min([xrange[0]] + epochs)
        xrange[1] = max([xrange[1]] + epochs)
        yrange[0] = min([yrange[0]] + train_accs + valid_accs)
        yrange[1] = max([yrange[1]] + train_accs + valid_accs)


    # Set axes titles
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")

    # set plot margins
    x_range = plot_params["x_range"]
    y_range = plot_params["y_range"]
    if x_range == "auto":
        x_range = (xrange[0], xrange[1])
    if y_range == "auto":
        y_range = (yrange[0], yrange[1])
    assert not isinstance(x_range, str)
    assert not isinstance(y_range, str)
    x_diff = x_range[1] - x_range[0]
    y_diff = y_range[1] - y_range[0]
    margin = 0.01
    x_range = (x_range[0]-margin*x_diff, x_range[1]+margin*x_diff)
    y_range = (y_range[0]-margin*y_diff, y_range[1]+margin*y_diff)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # If more than one label, sort them
    if len([l for l in labels if l != ""]) > 1:
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: pair[1],
                              reverse=True)
        handles, labels = zip(*sorted_pairs)
        plt.legend(handles, labels, prop={'family': 'monospace'})
        
    # Set grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set title and about to the ones stored in the last file.
    metadata,_ = _load_plot_data(json_paths[-1])
    title = "NO TITLE"
    if "title" in metadata:
        title = metadata["title"]
    fig.suptitle(title, fontsize=16)
    about = "NO ABOUT"
    if "about" in metadata:
        about:str = metadata["about"].rstrip()
    if common != "":
        about += " │ " + common
    ax.set_title(about, wrap=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.993))
    
    # export to image file
    plot_path = f"{path_prefix}plot.png"
    plt.savefig(plot_path, dpi=600)
    plt.close()
    _copy_to_dir(plot_path, _ALL_PLOTS_DIR_PATH)
    return

