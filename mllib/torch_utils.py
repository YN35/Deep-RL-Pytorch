import torch


# We avoid using torch.no_grad() while training (including evaluation while it) because it can badly affect on optimization using Adam (the reason is not clear... using detach() does not solve the issue)
# Use detach() or temporal_freeze() below intstead of no_grad().
def freeze(modules):
    for m in modules:
        m.train(False)
        for p in m.parameters():
            p.requires_grad = False

class temporal_freeze():
    def __init__(self, modules):
        self.modules = [(m, m.training) for m in modules]  # We assume that the top-level modules' training mode is the same for its children.
        params = []
        for m in modules:
            params.extend(m.parameters())
        self.params = [(p, p.requires_grad) for p in params]
    def __enter__(self):
        for m, _ in self.modules:
            m.train(False)
        for p, _ in self.params:
            p.requires_grad = False
    def __exit__(self, exc_type, exc_value, traceback):
        for m, f in self.modules:
            m.train(f)
        for p, f in self.params:
            p.requires_grad = f

def find_first_true(x, return_last_if_not_found=True):
    if return_last_if_not_found:
        x[:, -1] = True
    return torch.argmax(x.byte(), dim=-1, keepdim=True)  # argmax_cuda is not implemented for bool.