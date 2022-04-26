import torch.nn as nn

"""
Different layers normally used in torch
"""


class NonLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, activation=nn.ReLU()):
        super().__init__()
        self.__input_dim = input_dim
        self.__output_dim = output_dim
        self.bias = bias
        self.__sequential = [nn.Linear(self.__input_dim, self.__output_dim, bias=bias)]

        if activation is not None:
            self.__sequential.append(activation)
        self.__sequential = nn.Sequential(*self.__sequential)

    def forward(self, X):
        h = self.__sequential(X)

        return h
