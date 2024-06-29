import torch.utils.data as data

def collect_batch_data(**kwargs):
    assert 'rgb' in kwargs, 'rgb should be set for batch data'
    assert 'depth' in kwargs, 'depth should be set for batch data'
    return kwargs

class MyDataloader(data.Dataset):
    def __init__(self):
        raise (RuntimeError("__init__() is not implemented. "))

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        raise (RuntimeError("__getraw__() is not implemented."))

    def __getitem__(self, index):
        raise (RuntimeError("__getitem__() is not implemented."))

    def __len__(self):
        raise (RuntimeError("__len__() is not implemented."))