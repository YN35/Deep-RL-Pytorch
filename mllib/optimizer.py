import torch

def get_opt_obj(name):
    if name == 'AdamW':
        obj = torch.optim.AdamW
    else:
        assert KeyError
        
    return obj