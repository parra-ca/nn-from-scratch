#!/usr/bin/env python
from __future__ import annotations
from datetime import datetime
import sys

from .experiments import experiments as exp

list_of_experiments = [
    exp.fc_vs_cnn_on_mnist,
    exp.fc_vs_cnn_on_cifar,
]

if __name__ == "__main__":
    for one_experiment in list_of_experiments:
        one_experiment(
            do_train="train" in sys.argv,
            do_plot="plot" in sys.argv
        )
