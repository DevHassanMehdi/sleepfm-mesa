import os
import sys
import yaml
import json
import pickle
from typing import Any
import numpy as np

# Home directory on the account every pre-migration checkpoint/config.json
# was saved under. Historical saved configs bake in absolute paths rooted
# here; rebase_old_home_paths() rewrites them to the current account's
# home directory so old checkpoints keep working after an account migration.
_OLD_HOME_PREFIX = "/users/hamehdi/"


def rebase_old_home_paths(obj):
    """Recursively rewrite any string starting with _OLD_HOME_PREFIX to use
    the current account's home directory instead, including inside nested
    dicts/lists/tuples. Strings that don't start with the prefix (and any
    non-string value) are returned unchanged, so this is always safe to
    call on a config that was never affected."""
    if isinstance(obj, str):
        if obj.startswith(_OLD_HOME_PREFIX):
            return os.path.join(os.path.expanduser("~"), obj[len(_OLD_HOME_PREFIX):])
        return obj
    if isinstance(obj, dict):
        return {k: rebase_old_home_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        rebased = [rebase_old_home_paths(v) for v in obj]
        return type(obj)(rebased)
    return obj


def count_parameters(model):
    # Duck-typed on model.children()/.parameters()/.numel() -- never
    # references the torch module itself, so no import needed here.
    def count_recursive(module):
        total_params = 0
        num_layers = 0
        
        for child in module.children():
            child_layers, child_params = count_recursive(child)
            num_layers += child_layers
            total_params += child_params
        
        if list(module.children()) == []:  # if module has no children, it's a layer
            num_layers = 1
            for param in module.parameters():
                total_params += param.numel()
        
        return num_layers, total_params
    
    return count_recursive(model)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def instantiate_model(model_name, in_channel):
    model_class = getattr(sys.modules[__name__], model_name)
    return model_class(in_channel=in_channel)


def save_data(data: Any, filename: str) -> None:
    """
    Save data to a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - data: The data to save.
    - filename: The name of the file to save the data to. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    elif filename.endswith('.yaml'):
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    elif filename.endswith('.npy'):
        np.save(filename, data)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")


def load_data(filename: str) -> Any:
    """
    Load data from a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - filename: The name of the file to load the data from. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.

    Returns:
    - The loaded data.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")

def create_causal_mask(seq_len):
    import torch  # lazy: the only function in this module that needs torch
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return causal_mask
