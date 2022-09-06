import torch

def get_opt_func(name, params):
    if name == 'AdamW':
        for dic in params:
            opp = dic['opt_params']
            new_param = {'lr': opp.lr, 'weight_decay': opp.wd, 'betas': opp.adam_betas, 'eps': opp.adam_eps}
            dic.pop('opt_params')
            dic.update(new_param)
        obj = torch.optim.AdamW(params)
    else:
        obj = None
        assert KeyError
        
    return obj