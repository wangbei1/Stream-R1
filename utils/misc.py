import numpy as np
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            tensors = [d[k] for d in dict_list]
            if any(t.ndim == 0 for t in tensors):
                merged_dict[k] = torch.stack(tensors, dim=0)
            else:
                merged_dict[k] = torch.cat(tensors, dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict
