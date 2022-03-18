import sys
import inspect
import re

import numpy as np
import torch

from pyib.ml.models import Autoencoder

InputName = "../data/AutoencoderInput.npy"
OutputName= "../data/AutoencoderOutput.npy"
ModelName = "../data/Autoencoder.pt"

def generate_Autoencoder():
    # Specify the dimensions
    encoder_dim = [2]
    hidden_dim  = 1
    decoder_dim = [100,50,2]

    ac = Autoencoder(encoder_dim, hidden_dim, decoder_dim)

    # Random generate some data 
    randomData  = torch.rand(100,2)

    # Generate the output
    out = ac(randomData)

    np.save(InputName,randomData.numpy())
    np.save(OutputName,out.detach().numpy())
    torch.save(ac.state_dict(), ModelName)

def test_Autoencoder():
    # Specify the dimensions
    encoder_dim = [2]
    hidden_dim  = 1
    decoder_Dim = [100,50,2]

    # Load data
    randomData = torch.tensor(np.load(InputName).astype(np.float32))
    refres     = torch.tensor(np.load(OutputName).astype(np.float32))

    # Create 
    ac  = Autoencoder(encoder_dim, hidden_dim, decoder_Dim)
    ac.load_state_dict(torch.load(ModelName))

    # Randomly generate some data
    res = ac(randomData)

    # assert all close
    assert torch.allclose(refres, res)


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^generate_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
