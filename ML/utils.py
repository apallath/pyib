import torch
import numpy as np

def prep_data_PIB(file:str):
    """
    Function that preps the data for PIB reading from a file, the file will have the following format 
    ' # time x1 x2 x3 ... '
    
    where the xi are the CVs collected from the simulation

    Input:
    -----
        file(str)       : The input .dat file as a str --> could be a npy file as well
    """
    file_split = file.split(".")