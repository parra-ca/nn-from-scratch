from __future__ import annotations
from typing import Any
import os, torch, traceback
from datetime import datetime

from . import models
from . import dataloaders as dld
from . import plotter

torch.set_printoptions(precision=3, sci_mode=False, linewidth=512)
FLOAT_DTYPE = torch.float32
ACCEL_DEV = "cpu"
RESULTS_DIR_BASEPATH = "results"

# ------------------------------------------------------------------------
# Utility functions.
# ------------------------------------------------------------------------
def _set_accelerator(device_str: str) -> None:
    global ACCEL_DEV
    if device_str == "xpu":
        if torch.xpu.is_available():
            ACCEL_DEV = torch.device(device_str)
        else:
            raise RuntimeError("XPU not available")
        
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            ACCEL_DEV = torch.device(device_str)
        else:
            raise RuntimeError("CUDA not available")
    
    elif device_str == "mps":
        if torch.backends.mps.is_available():
            ACCEL_DEV = torch.device(device_str)
        else:
            raise RuntimeError("MPS not available")

    elif device_str in ("auto", "cpu"):
        cur_dev = torch.accelerator.current_accelerator(check_available=True)
        if cur_dev:
            ACCEL_DEV = cur_dev
        else:
            ACCEL_DEV = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_str}")

def _random_seed(seed: int = 42) -> None:
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
        do_save_params: bool = True,
        do_save_plot_data: bool = True,
):
    """Load data, train a model, evaluate on test set, and save results.
      general
          Experiment metadata. Keys: "title", "about", "dir" (results subdirectory name).
    
      data_params
          Dataset configuration. Keys: "dataset" (dataloader class)
          "sample", "sample_dir", "augm_params", "val_split" (proportion of training
          data reserved for validation), "batch_sz".

      hyp_params : model and training hyperparameters.
          The only required keys are
            - "model"     : your Model subclass
            - "model_fns" : dict with "loss": Callable[[Tensor,Tensor],Tensor] and
                            "decision": Callable[[Tensor],Tensor]
                            it may have other classes or functions used by your
                            Model subclass.
            - "load_path" : if you are loading parameters from a file.
          The rest are hyperparameters that you may want to use in your 
          Model.__init__(), .forward(), and .fit().
          For example: "epochs", "lrn_rate", "w_decay", "betas", "dropout".
    
      plot_params: options concerning the data plotting
          - "keys_general"
            "keys_data"
            "keys_hparam" : items from the other dicts to save in the plot files
          - "size"        : size of the plot. tuple[int, int]
          - "x_range"     : "auto" or tuple[int, int],
          - "y_range"     : "auto" or tuple[int, int],
          - "plot_train"
            "plot_valid"
            "plot_test"   : plot the training/validation/test accuracy lines?

      do_save_params: save best model parameters to a .params file?

      do_save_plot_data: save metadata and accuracy log to a json file?
    """
    _random_seed(general["seed"])
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    
    # DETECT ACCELERATOR DEVICE
    _set_accelerator(hyp_params["device"])
    print("\nACCELERATOR:", str(ACCEL_DEV).upper())

    
    # LOAD DATA
    print(f"\nLOADING DATA")
    dataloader = data_params["dataloader"]
    if not issubclass(dataloader, dld.Loader):
        print("The given dataloader is not a subclass of dataloaders.Loader.")
        return
    data_source = dataloader.create(data_params)
    data_source.describe()

    
    # EXPORT DATA SAMPLE
    if data_params["sample"] > 0:
        print(f"\nEXPORTING DATA SAMPLE")
        sample_count = data_params["sample"]
        sample_dir = data_source.export_sample(sample_count)
        print(f"{sample_count} samples exported to {sample_dir}.")
        return

    
    # CREATE MODEL, LOAD PARAMETERS, MOVE TO ACCELERATOR
    NeuralNetModel = hyp_params["model"]
    if not issubclass(NeuralNetModel, models.Model):
        print("The given model is not a subclass of models.Model.")
        return
    load_params_path = hyp_params["load_path"]
    print("\nCREATING MODEL")
    in_shape = torch.tensor(data_source.shape)
    out_shape = torch.tensor(len(data_source.labels))
    model = NeuralNetModel(hyp_params, in_shape, out_shape)
    model.describe()
    if isinstance(load_params_path, str):
        print(f"LOADING PARAMS: {load_params_path}")
        model.load_state_dict(torch.load(load_params_path, weights_only=True))
    model.to(ACCEL_DEV)

    
    # TRAIN
    user_interrupted = False
    try:
        print(f"\nTRAINING")
        model.fit(data_source)
    except KeyboardInterrupt:
        print("\n\n---- INTERRUPTED! ----\n")
        general["TRUNCATED"] = True
        user_interrupted = True
    except Exception as e:
        traceback.print_exc()
        print("\n\n---- EXCEPTION! ----\n")
        general["EXCEPTION"] = True


    # EVALUATE TEST DATASET
    test_corr, test_tot = model.evaluate(data_source.dl_test)
    test_accu = 100 * test_corr / test_tot

    
    # DEFINE SAVE PATH STUB
    result_dir = general["result_dir"]
    result_dir = os.path.join(RESULTS_DIR_BASEPATH, result_dir)
    os.makedirs(result_dir, exist_ok=True)
    filename = f"{test_accu:06.2f}__{timestamp}"
    save_stub = str(os.path.join(result_dir, filename))
    for k in ["EXCEPTION", "TRUNCATED"]:
        if k in general:
            save_stub += f"_{k}"

    
    # SAVE MODEL PARAMETERS
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
        plotter.save_plot_data(save_plot_path,
                               metadata,
                               accuracy)
    return user_interrupted
