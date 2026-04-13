from __future__ import annotations
import math
from typing import Mapping, Any, override
import torch
from torch import Tensor, nn

from .. import util
from .. import dataloaders as dld
from . import common,fc

class ConvolutionalNet(common.Model):
    """Convolutional network with batch normalization and MaxPool at each block.
    AvgPool and fully connected head with optional dropout."""
    def __init__(
            self,
            hyp_params: Mapping[str, Any],
            in_shape: Tensor,
            out_shape: Tensor,
            do_init: bool = True,
    ):
        super().__init__(hyp_params)

        # convolution blocks
        self.conv_blocks: nn.ModuleList = nn.ModuleList()
        # [(out_ch, ker_sz, stride, pool_sz), ...]
        conv_cksp: list[tuple[int, int, int, int]] = hyp_params["conv_cksp"]
        conv_activ_hl = hyp_params["model_fns"]["conv_activ_hl"]
        in_ch = int(in_shape[0])
        for out_ch, ker_sz, stride, pool_sz in conv_cksp:
            mods: list[nn.Module] = []
            padding = 0 if stride > 1 else "same"
            mods.append(nn.Conv2d(
                in_ch, out_ch, kernel_size=ker_sz, padding=padding,
                stride=stride, dtype=util.FLOAT_DTYPE,
                bias=False)) # normalization does the shifting
            mods.append(nn.BatchNorm2d(
                out_ch, dtype=util.FLOAT_DTYPE))
            mods.append(conv_activ_hl())
            mods.append(nn.MaxPool2d(pool_sz)) if pool_sz > 1 else None
            self.conv_blocks.append(nn.Sequential(*mods))
            in_ch = out_ch

        
        # global average pooling or simply flatten
        if hyp_params["to_head"] == "gap":
            self.to_fc_input = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            head_in_shape = torch.tensor([in_ch])
        elif hyp_params["to_head"] == "flatten":
            self.to_fc_input = nn.Flatten(1)
            dummy = torch.zeros([1, *in_shape.tolist()])
            for block in self.conv_blocks:
                dummy = block(dummy)
            head_in_shape = torch.tensor([math.prod([*dummy.shape])])
        else:
            raise ValueError("Unrecognized hyp_params['to_head']: "
                             f"'{hyp_params['to_head']}'")

        
        # Fully connected head (all but last layer)
        head_params = {
            "model_fns" : {
                "activ_hl": hyp_params["model_fns"]["head_activ_hl"],
                "decision": hyp_params["model_fns"]["decision"],
                "loss"    : hyp_params["model_fns"]["loss"],
            },
            "hid_layers": hyp_params["head_hid_lys"],
            "dropout" : hyp_params.get("dropout"), # maybe None
        }
        head_out_shape = out_shape
        self.head = fc.FullyConnectedNet(head_params, head_in_shape,
                                      head_out_shape, do_init=False)

        self.apply(self._init_weights) if do_init else None

        return

    def _init_weights(self, mod: nn.Module):
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
            # bias disabled for convolution layers
        elif isinstance(mod, nn.BatchNorm2d):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
        return
    
    @override
    def forward(self, input_batch: Tensor) -> Tensor:
        a = input_batch
        # convolution blocks
        for block in self.conv_blocks:
            a = block(a)
            
        # convert conv output to flat layer
        a = self.to_fc_input(a)
        
        # fully connected head
        return self.head(a)

    @override
    def fit(self, data_source: dld.Loader) -> None:
        common.fit_loop_adamw_cosine(self, data_source)
        return
