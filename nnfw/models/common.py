from __future__ import annotations
from typing import Callable, Mapping, Any, cast, Sized
import torch, copy
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .. import util
from .. import dataloaders as dld

# percentage drop considered a network collapse
_COLLAPSE_THRESHOLD = 50

# ------------------------------------------------------------------------
# Base module. User needs to implement __init__, forward and fit methods.
# Hyper-parameters to be used by any of those methods must be contained
# in the 'hyp_params' dictionary.
# ------------------------------------------------------------------------
class Model(nn.Module):
    """Base class for neural network models.
    
    Subclasses must implement:
        forward(input_batch) -> Tensor
        fit(data_source) -> None
    
    Provides: evaluate, monitor_accuracy, describe.
    """
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
    ):
        super().__init__()
        
        self.fn_decision: Callable[[Tensor], Tensor] = \
            hyp_params["model_fns"]["decision"]
        self.fn_loss: Callable[[Tensor,Tensor], Tensor] = \
            hyp_params["model_fns"]["loss"]
        self.hyp_params = hyp_params
        
        # training state and log
        self.best_state_dict: dict[str, Tensor] | None = None
        self.best_valid_acc: float = 0.0
        self.last_valid_acc: float = 0.0
        self.accuracy_log: list[tuple[float,float]] = []
        return

    @classmethod
    def create(cls,
               hyp_params: Mapping[str, Any],
               in_shape: Tensor,
               out_shape: Tensor,
               do_init: bool = True,            
    ):
        raise NotImplementedError("Subclasses must implement create()")

    def forward(self, input_batch: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def fit(self, data_source: dld.Loader) -> None:
        raise NotImplementedError("Subclasses must implement fit()")

    def evaluate(self, data_loader: DataLoader) -> tuple[int, int]:
        """Predicts the output for each sample in the data_loader.
        Returns a tuple with the count of correct and total samples.
        """
        was_training = self.training
        self.eval()

        # evaluate in batches
        correct_count = torch.zeros(1, dtype=torch.int64, device=util.ACCEL_DEV)
        with torch.no_grad():
            for X_batch,Y_batch in data_loader:
                X_batch = X_batch.to(util.ACCEL_DEV)
                Y_batch = Y_batch.to(util.ACCEL_DEV)
                out_layer = self(X_batch)
                y_hat_batch = self.fn_decision(out_layer)                
                correct_count += (y_hat_batch == Y_batch).sum()
        if was_training:
            self.train()
        dataset_sz = len(cast(Sized, data_loader.dataset))
        return (int(correct_count), dataset_sz)

    def monitor_accuracy(self,
                         dl_train: DataLoader,
                         dl_valid: DataLoader,
                         save_best: bool = False,
                         ) -> tuple[str, bool]:
        """Evaluate, print, and keep registry of model's accuracy.

        Register and prints train and validation accuracy.
        Checkpoint parameters if best performing seen so far.
        Early exit if validation accuracy is equal-ish to 100%.
        Early exit if the model collapsed (massive drop from last monitoring)
        """
        # evaluate accuracy for training/validation samples
        train_correct, train_total = self.evaluate(dl_train)
        valid_correct, valid_total = self.evaluate(dl_valid)
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
            elif isinstance(p, (tuple, list)):
                to_print.append((n,str(p).replace(" ", "")))
            elif isinstance(p, type):
                to_print.append((n,p.__name__))
        name_width = max(len(n) for n,_ in to_print)
        for n,p in to_print:
            print(f"  {n:{name_width}} : {str(p)}")
        return

def argmax(a_batch: torch.Tensor) -> torch.Tensor:
    return torch.argmax(a_batch, dim=1)

def fit_loop_adamw_cosine(model: Model, data_source: dld.Loader) -> None:
    """AdamW training loop with:
    - weight decay in weights only.
    - cosine annealing LR scheduling.
    - train/validation performance monitoring
    - checkpoint to best validation performance
    """
    epochs = model.hyp_params["epochs"]
    weight_decay = model.hyp_params["w_decay"]
    betas = model.hyp_params["betas"]
    learning_rate = model.hyp_params["lr"]
    lr_min = model.hyp_params["lr_min"]

    # create optimizer
    model.train()
    optimizer = torch.optim.AdamW(
        params=[
            {"params":[p for n,p in model.named_parameters() if 'weight' in n],
             "weight_decay": weight_decay},
            {"params":[p for n,p in model.named_parameters() if 'bias' in n],
             "weight_decay": 0.0}
        ],
        lr=learning_rate,
        betas=betas,
    )

    # create LR scheduler
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min)

    # train a number of epochs
    for epoch in range(epochs):
        print(f"Epoch {(epoch+1):3}: ", end="", flush=True)
        
        # do one gradient step per mini_batch
        for X_batch,Y_batch in data_source.dl_train:
            X_batch = data_source.augment(X_batch)
            X_batch = X_batch.to(util.ACCEL_DEV) # move to GPU after transform
            Y_batch = Y_batch.to(util.ACCEL_DEV)
            optimizer.zero_grad()
            out_ly_batch = model(X_batch)
            loss = model.fn_loss(out_ly_batch, Y_batch)
            loss.backward()
            optimizer.step()

        # update learning rate
        sched_lr.step()
        
        # monitor network accuracy
        accu_msg, do_break = model.monitor_accuracy(
            data_source.dl_tr_sample,
            data_source.dl_valid,
            save_best=True
        )
        print(accu_msg)
        if do_break:
            break
        
    # restore parameters of the best performing net
    model.eval()
    if model.best_state_dict is not None:
        model.load_state_dict(model.best_state_dict)
    else:
        raise RuntimeError("No best state to restore")
    return
