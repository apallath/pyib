from ML.Autoencoder import Autoencoder
import numpy as np
import torch

def testAutoencoder():
    # specify the dimensions
    dim = [2,128,128,2]

    ac  = Autoencoder(dim)

    # Randomly generate some data
    randomData = torch.rand(100,2)
    res = ac(randomData)

    assert res.shape[1] == dim[-1]

if __name__ == "__main__":
    testAutoencoder()