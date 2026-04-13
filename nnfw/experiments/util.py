from __future__ import annotations
from typing import Any

from ..util import model_train_eval
from ..plotter import plot_all

def _new_reset_dict(base, override) -> dict[str, Any]:
    new = dict(**base)
    new.update(override)
    return new

def _update_dicts(hyper, data, plot, var):
    for d,key,val in var:
        if d == "h":
            hyper[key] = val
        elif d == "d":
            data[key] = val
        elif d == "p":
            plot[key] = val
    return

def run_experiments(hyper_base, hyper_override,
                    data_base, data_override,
                    plot_base, plot_override,
                    general, variations,
                    do_train, do_plot):
    for var in variations:
        hyper = _new_reset_dict(hyper_base, hyper_override)
        data = _new_reset_dict(data_base, data_override)    
        plot = _new_reset_dict(plot_base, plot_override)
        _update_dicts(hyper, data, plot, var)

        user_interruped = False
        if do_train:
            user_interruped = model_train_eval(general, data, hyper, plot)
        if do_plot:
            plot_all(plot, general["result_dir"])
            if not do_train:
                break
        if user_interruped:
            exit(0)
    return
