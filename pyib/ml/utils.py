import torch
import numpy as np

def prep_data_PIB(file:str, dt:int):
    """
    Function that preps the data for PIB reading from a file, the file will have the following format 
    ' # time x1 x2 x3 ... '
    
    where the xi are the CVs collected from the simulation

    Input:
    -----
        file(str)       : The input .dat file as a str --> could be a npy file as well
        dt(int)         : The time step lag 
    """
    def parseNPY(file:str):
        data = np.load(file)

        return data

    def parseDAT(file:str):
        f = open(file, "r")
        lines = np.array([[float(f) for f in line.nstrip("\n").split()] for line in f.readlines()])

        return lines

    # register the functions into a dictionary
    loadDic_   = {
                "npy" : parseNPY,
                "dat" : parseDAT
    }

    # automatically read the type of files
    file_split = file.split(".")
    filetype   = file_split[1]

    # load the parsing function
    parseFunc  = loadDic_[filetype]
    parsedData = parseFunc(file)[:,1:]

    # Parsed data will be [time, x1, x2, x3, ... , ]
    # We will make an X that has [[x1(t),x2(t),x3(t), ..],...] and y that has [[x1(t+dt), x2(t+dt), x3(t+dt), ...],...]
    X = torch.tensor(parsedData[:,:-dt].astype(np.float32))
    y = torch.tensor(parsedData[:,dt:].astype(np.float32))

    assert len(X) == len(y) , "The length of X and y do not agree, something is wrong with the code."

    return X, y

def Normalize_data(X:torch.tensor, axis=None):
    """
    Normalize the data along dimension dim
    """
    if axis is not None:
        X_norm = (X - X.mean(axis=axis))/X.std(axis=axis)
    else:
        X_norm = (X - X.mean())/X.std()
    
    return X_norm