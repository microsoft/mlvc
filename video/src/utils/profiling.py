# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import logging
from torch import nn
from collections import defaultdict
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        class_name = args[0].__class__.__name__
        method_name = func.__name__

        logging.debug(f"Time for {class_name}.{method_name} --- {round((end_time - start_time) * 1000, 4)} ms")

        return result

    return wrapper


def summarize_model_layers(model: nn.Module):
    # Initialize a dictionary to store the sum of parameters for each root name
    param_sum_by_module = defaultdict(int)

    # Iterate through each submodule
    for submodule_name, submodule in model.named_modules():
        # Iterate through parameters of each submodule
        for parameter in submodule.parameters(recurse=False):
            param_sum_by_module[submodule_name] += parameter.numel()

    sorted_param_sum_by_module = sorted(param_sum_by_module.items(), key=lambda x: x[1], reverse=True)

    for module_name, sum_params in sorted_param_sum_by_module:
        logging.info(f"Module Name: {module_name}, Total Parameters: {sum_params}")

    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
