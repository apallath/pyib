import sys
import inspect
import re

import numpy as np
import torch

from pyib.ml.models import Autoencoder


def test_Autoencoder():
    # Specify the dimensions
    encoder_dim = [2]
    hidden_dim  = 1
    decoder_Dim = [100,50,2]

    ac  = Autoencoder(encoder_dim, hidden_dim, decoder_Dim)

    # Randomly generate some data
    randomData = torch.rand(100,2)
    res = ac(randomData)

    assert res.shape[1] == 2


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
