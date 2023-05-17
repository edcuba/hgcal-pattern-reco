import torch
import numpy as np


class DummyModel:
    def eval(self):
        pass

class DummyPleaser(DummyModel):

    def __init__(self, output_dim=1) -> None:
        super(DummyPleaser).__init__()
        self.output_dim = output_dim

    def __call__(self, X, edge_list=None):

        out_size = len(X)
        if edge_list is not None:
            out_size = len(edge_list)

        if self.output_dim == 1:
            return torch.tensor(np.ones(out_size))

        return torch.tensor(np.ones((out_size, self.output_dim)))

class DummyGuesser(DummyModel):
    def __call__(self, X):
        return torch.tensor(np.random.random(len(X)))

class DummyScaler:
    def transform(self, X):
        return X.tolist()

class GraphNaiveDummy(DummyModel):
    def __call__(self, _, edge_list):
        return torch.tensor(np.ones(edge_list.shape[1])).type(torch.float)

class GraphRandomDummy(DummyModel):
    def __call__(self, _, edge_list):
        return torch.tensor(np.random.random(edge_list.shape[1])).type(torch.float)
