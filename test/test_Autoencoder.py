from ML.Autoencoder import Autoencoder
import numpy as np
import torch

def testAutoencoder():
    # specify the dimensions
    encoder_dim = [2]
    hidden_dim  = 1
    decoder_Dim = [100,50,2]

    ac  = Autoencoder(encoder_dim, hidden_dim, decoder_Dim)

    # Randomly generate some data
    randomData = torch.rand(100,2)
    res = ac(randomData)

    assert res.shape[1] == 2


if __name__ == "__main__":
    testAutoencoder()
