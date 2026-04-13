from __future__ import annotations
from typing import Mapping, Any, override
import torch
from torch import Tensor, nn

from .. import util
from .. import dataloaders as dld
from . import common

class FullyConnectedNet(common.Model):
    """Feedforward netwrk with optional dropout."""
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
            in_shape: Tensor,
            out_shape: Tensor,
            do_init: bool = True,            
    ):
        super().__init__(hyp_params)

        # if defined, include dropout
        dropout = hyp_params.get("dropout")

        # fully connected sequence of layers with ReLU activation
        self.layers: nn.ModuleList = nn.ModuleList()
        hid_layers = hyp_params["hid_layers"]
        hid_activ_fn = hyp_params["model_fns"]["activ_hl"]
        fan_in = int(torch.prod(in_shape).item())
        for fan_out in hid_layers:
            mods: list[nn.Module] = []
            mods.append(nn.Linear(fan_in, fan_out, bias=True,
                                    dtype=util.FLOAT_DTYPE))
            mods.append(hid_activ_fn())
            mods.append(nn.Dropout(p=dropout)) if dropout else None
            self.layers.append(nn.Sequential(*mods))
            fan_in = fan_out

        # last layer.
        fan_out = int(torch.prod(out_shape).item())
        mods: list[nn.Module] = []
        mods.append(nn.Linear(fan_in, fan_out, bias=True,
                                dtype=util.FLOAT_DTYPE))
        self.layers.append(nn.Sequential(*mods))

        
        self.apply(self._init_weights) if do_init else None
        return

    def _init_weights(self, mod: nn.Module):
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
        return

    @override
    def forward(self, input_batch: Tensor) -> Tensor:
        a = input_batch.flatten(1)
        for layer in self.layers:
            a = layer(a)
        return a

    @override
    def fit(self, data_source: dld.Loader) -> None:
        common.fit_loop_adamw_cosine(self, data_source)
        return
