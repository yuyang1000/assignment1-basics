import torch

DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DEVICE


def to_device(tensor):
    device = get_device()
    if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
        tensor = tensor.to(device)
    return tensor