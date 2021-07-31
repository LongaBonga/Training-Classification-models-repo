import torch

AVAILABLE_OPTIMS = ['sgd', 'rmsprop', 'adam', 'adadelta']

def build_optimizer(net, optimizer_name):
    assert optimizer_name in AVAILABLE_OPTIMS
    if optimizer_name == 'adadelta':
        optim = torch.optim.Adadelta(net.parameters(), lr=1e-3)
    elif optimizer_name == 'adam':
        optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    elif optimizer_name == 'rmsprop':
        optim = torch.optim.RMSprop(net.parameters(), lr=1e-3)
    elif optimizer_name == 'sgd':
        optim = torch.optim.SGD(net.parameters(), lr=1e-3)

    return optim
