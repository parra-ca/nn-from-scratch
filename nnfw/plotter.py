from __future__ import annotations
import pathlib, os, json, shutil, textwrap, glob
from typing import Any
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from . import util

_ALL_PLOTS_DIR_PATH = "plots"
_LABEL_CH_WIDTH = 100

# ------------------------------------------------------------------------
# Plotting Functions
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
                is_common = False # always goes in legend
                v_wd = 5
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
        label_parts = []
        # first the shared key with diff value, then unique
        for k in keys_diff_val+keys_unique:
            if k not in meta:
                continue
            if k in error_keys:
                label_parts.append(f"{k}")
                continue
            v, w = meta[k], vals_width[k]
            if k == "test_accu":
                v_str = f"{v:>05.{w-3}f}%"
            elif isinstance(v, float):
                v_str = f"{v:<{w}}"
            elif isinstance(v, int):
                v_str = f"{v:>{w}}"
            else:
                v_str = f"{v:<{w}}"
            label_parts.append(f"{k}:{v_str}")
            
        labels.append(" │ ".join(label_parts))

    # put together all the common parameters
    common = " │ ".join(f"{k}:{meta_per_file[0][k]}" for k in keys_common)

    return labels, common

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
    path_for_name = src_path.removeprefix(util.RESULTS_DIR_BASEPATH+"/")
    unique_name = "_".join(pathlib.Path(path_for_name).parts)
    shutil.copy2(src_path, os.path.join(dest_dir,unique_name))
    return

def save_plot_data(path: str,
                   metadata: dict[str, Any],
                   accuracy: dict[str, Any],
                   ) -> None:
    """Save json file with the model's hyperparameters and its train
    and validation accuracy at each epoch; plus its final test accuracy.
    
    - metadata: dictionary with the hyper-parameters to save.
    - accuracy: dictionary with train and validation accuracy series,
                plus test accuracy scalar.

    This file can then be loaded with '_load_plot_data()' and plotted
    with function 'plot_all()'
    """
    payload = {
        "metadata" : metadata,
        "accuracy" : accuracy
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return

def export_sample(imgs: tuple[Tensor,Tensor] | tuple[Tensor], filename: str, title: str,
                  cmap_val: str | None, dpi: int = 75) -> None:
    if len(imgs) == 1: # single image
        fig,ax0 = plt.subplots(1, 1, figsize=(4,4))
    elif len(imgs) == 2: # side-by-side images
        fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
        ax1.imshow(imgs[1], cmap=cmap_val)
        ax1.set_title("(augmented)")
        ax1.axis("off")
    else:
        raise ValueError(f"Unexpected number of samples to export {len(imgs)}")
    
    ax0.imshow(imgs[0], cmap=cmap_val)
    ax0.axis("off")
    ax0.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return

def plot_all(
        plot_params: dict[str, Any],
        subdir_path: str,
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
    results_path = os.path.join(util.RESULTS_DIR_BASEPATH, subdir_path)
    print(f"\nPLOTTING ALL JSON FILES IN '{results_path}'.")
    json_paths = sorted(glob.glob(os.path.join(results_path, '*.json')),
                        key=os.path.getmtime)
    if not json_paths:
        print(f"No '{results_path}/*.json' files found.")
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
        if len(epochs) == 0:
            print(f"File '{path}' has no data to plot.")
            continue
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
            if not plot_params["plot_test"]:
                alt_label = f"val*:{best_val_accu:5.2f} | " + str(label)
            else:
                alt_label =  None
            ax.plot(epochs, valid_accs,
                    linestyle="-",
                    linewidth=1.33,
                    color=color,
                    alpha=0.6,
                    zorder=2,
                    label=alt_label,
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
        xrange[0] = min(xrange[0], epochs[0])
        xrange[1] = max(xrange[1], epochs[-1])
        yrange[0] = min([yrange[0]] + train_accs + valid_accs)
        yrange[1] = max([yrange[1]] + train_accs + valid_accs)

    # check if any file had data to plot
    if float("-inf") in xrange + yrange or \
       float("inf")  in xrange + yrange or \
       xrange[0] == xrange[-1]:
        print(f"No file had data!")
        return
    
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

    # If there are labels, sort them.
    if len([l for l in labels if l != ""]) > 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: pair[1],
                              reverse=True)
        handles, labels = zip(*sorted_pairs)
        plt.legend(handles, labels, prop={'family': 'monospace'})
        
    # Set grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.3)
    ax.grid(visible=True, which="minor", linestyle=":", alpha=0.3, linewidth=0.5)

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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.980))
    
    # export to image file
    plot_path = os.path.join(results_path, "plot.png")
    plt.savefig(plot_path, dpi=600)
    plt.close()
    _copy_to_dir(plot_path, _ALL_PLOTS_DIR_PATH)
    return

