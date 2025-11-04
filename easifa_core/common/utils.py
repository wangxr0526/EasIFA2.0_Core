"""
Utility functions for EasIFA inference.
"""
import os
import torch
import yaml
import dgl
import numpy as np
from Bio import pairwise2
from transformers.tokenization_utils_base import BatchEncoding




def colorize(text, color):
    """Colorize terminal text output."""
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'cyan': '36',
        'reset': '0',
        'bold': '1',
    }
    return f"\033[{colors[color]}m{text}\033[0m"


def read_model_state(model_save_path):
    """
    Read model state and arguments from checkpoint directory.
    
    Args:
        model_save_path: Path to checkpoint directory
        
    Returns:
        model_state: Model state dict
        args: Model arguments dict
    """
    model_state_fname = os.path.join(model_save_path, 'model.pth')
    args_fname = os.path.join(model_save_path, 'args.yml')

    model_state = torch.load(model_state_fname, map_location=torch.device('cpu'))
    keys = list(model_state.keys())
    if keys and 'module.' in keys[0]:
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    args = yaml.load(open(args_fname, "r"), Loader=yaml.FullLoader)

    return model_state, args


def load_pretrain_model_state(model, pretrained_state, load_active_net=True, viz_warning=True):
    """
    Load pretrained model state into model.
    
    Args:
        model: Model to load state into
        pretrained_state: Pretrained state dict
        load_active_net: Whether to load active_net layers
        viz_warning: Whether to print warning messages
        
    Returns:
        model: Model with loaded state
    """
    model_state = model.state_dict()
    pretrained_state_filter = {}
    extra_layers = []
    different_shape_layers = []
    need_train_layers = []
    
    for name, parameter in pretrained_state.items():
        if name in model_state and parameter.size() == model_state[name].size():
            pretrained_state_filter[name] = parameter
        elif name not in model_state:
            extra_layers.append(name)
        elif parameter.size() != model_state[name].size():
            different_shape_layers.append(name)
    
    if not load_active_net:
        for name in list(pretrained_state_filter.keys()):
            if 'active_net' in name:
                del pretrained_state_filter[name]
    
    for name, parameter in model_state.items():
        if name not in pretrained_state_filter:
            need_train_layers.append(name)

    model_state.update(pretrained_state_filter)
    model.load_state_dict(model_state)
    
    if viz_warning:
        print('Extra layers:', extra_layers)
        print('Different shape layers:', different_shape_layers)
        print('Need to train layers:', need_train_layers)
    
    return model


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    
    Args:
        obj: Object to transfer (tensor, dict, list, tuple, DGLGraph, etc.)
        
    Returns:
        Object transferred to CUDA device
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, dgl.DGLGraph):
        return obj.to(*args, **kwargs)
    elif isinstance(obj, BatchEncoding):
        return obj.to(*args, **kwargs)

    raise TypeError("Can't transfer object type `%s`" % type(obj))
def get_active_site_binary(active_site, aa_sequence_len, begain_zero=True):

    active_site_bin = np.zeros((aa_sequence_len,))
    for one_site in active_site:
        if len(one_site) == 1:
            if begain_zero:
                active_site_bin[one_site[0]] = 1
            else:
                active_site_bin[one_site[0] - 1] = 1
        elif len(one_site) == 2:
            b, e = one_site
            if begain_zero:
                # site_indices = [k for k in range(b, e)]
                site_indices = [k for k in range(b, e + 1)]
            else:
                # site_indices = [k-1 for k in range(b, e)]
                site_indices = [k - 1 for k in range(b, e + 1)]
            active_site_bin[site_indices] = 1
        else:
            raise ValueError("The label of active site is not standard !!!")
    return active_site_bin

def map_active_site_for_one(seqA, seqB, active_site_A, begain_zero=True):
    seqA_len = len(seqA)
    active_site_A_bin = get_active_site_binary(active_site_A, seqA_len, begain_zero)
    alignments = pairwise2.align.localxx(seqB, seqA)  # 取最佳结果
    if alignments:
        alignment = alignments[0]
        padded_seqA = alignment[1]
        padded_seqB = alignment[0]
        padded_seqB_arr = np.asanyarray([x for x in padded_seqB])
        padded_active_site_A_bin = np.zeros((len(padded_seqA),))
        org_idx = 0
        for padded_idx, aa in enumerate(padded_seqA):
            if aa != "-":
                padded_active_site_A_bin[padded_idx] = active_site_A_bin[org_idx]
                org_idx += 1
            else:
                pass
        seqB_active_site_bin = padded_active_site_A_bin[padded_seqB_arr != "-"]
    else:
        seqB_active_site_bin = None
    return seqB_active_site_bin
