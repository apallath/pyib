### Training utility functions ###
import torch
from .models import VAE
from ..md.utils import TrajectoryReader
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

class Loader(torch.utils.data.Dataset):
    """
    Args:
        filename(str)   : The file name of the input trajectory
        generate_label_func(callable)   : A function that generates the label for trajectory X (N,d1)
        delta_t(int)    : The lag time dt
    """
    def __init__(self, X:torch.tensor, labels:torch.tensor) -> None:
        # Convert trajectory to tensor 
        self.X  = X

        # Generates a label of shape (N,d2)
        self.y = labels
    
    def __len__(self):
        return len(self.X) 
    
    def __getitem__(self,index):
        """
        Returns the input X and time index t, and y at time index t+dt

        Args:
            index(int)      : The input time index (time step)
        """
        return self.X[index], self.y[index]


def SPIB_data_prep(file_name:str, dt:int, label_func:callable, train_percent=0.8,comment_str="#", format="txyz", skip=1):
    """
    Function that prepares the data for SPIB training 

    Args:
        train_percent(float)    : The percent of data used for training 
    """
    traj_reader = TrajectoryReader(file_name, comment_char=comment_str, format=format)

    # read the trajectory
    time, traj = traj_reader.read_traj(skip=skip)
    traj = torch.tensor(traj, dtype=torch.float32)
    N = len(traj) - dt
    num_train = int(N * train_percent)

    # Select the training points
    randperm = torch.randperm(N)
    trainIdx  = randperm[:num_train]
    testIdx   = randperm[num_train:]

    # Create labels
    label = label_func(traj)

    # split train/test data/labels
    train_traj = traj[trainIdx,:2]
    train_labels = label[trainIdx + dt]
    test_traj  = traj[testIdx, :2]
    test_labels = label[testIdx + dt]

    return train_traj, train_labels, test_traj, test_labels


def SPIB_train(VAE_model:VAE, filename:str, labels_func:callable, dt:int, lr=1e-3, update_labels=True, update_labels_freq=10, batch_size=1028, epochs=1000, \
    skip=1, beta=0.01, print_every=-1, threshold=0.01, device='cpu', comment_str="#", format="txyz"):
    """
    Function that performs the SPIB training process

    Args:
        VAE_model(torch.nn.Module)  : A torch model for beta-VAE
        filename(str)               : The filename of the trajectory input
        labels(torch.tensor)        : Torch tensor for output, shape (N,d2)
        dt(int)                     : The time lag used in the system
        lr(float)                   : The learning rate applied for the optimizer 
        update_labels(bool)         : Whether or not we are updating the labels at various epochs
    """
    # define optimizer 
    optimizer = optim.Adam(VAE_model.parameters(), lr=lr)

    # Define the losses 
    CrossEntropyLoss = nn.CrossEntropyLoss()

    # Prepare input data 
    TrainX, TrainY, testX, testY = SPIB_data_prep(filename, dt, labels_func, skip=skip, comment_str=comment_str, format=format)

    # Move data to device (could be GPU)
    TrainX = TrainX.to(device)
    TrainY = TrainY.to(device)
    testX  = testX.to(device)
    testY  = testY.to(device)

    TrainX = (TrainX - TrainX.mean(axis=0))/TrainX.std(axis=0)
    VAE_model.to(device)

    # Define dataset, trainloader and testloader
    TrainDataset = Loader(TrainX, TrainY)
    TestDataset  = Loader(testX, testY)
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    TestLoader  = DataLoader(TestDataset, batch_size=batch_size)

    # before training, track the state population
    _, state_population0 = VAE_model.get_Labels(TrainX)

    for i in range(epochs):
        avg_epoch_loss = 0
        avg_KL_loss    = 0
        avg_CE_loss    = 0

        for X, y in TrainLoader:
            # Clear the gradient in optimizer
            optimizer.zero_grad()

            # obtain the y prediction
            y_pred, mean, logvar, sampled_z = VAE_model(X)

            CELoss = CrossEntropyLoss(y_pred, y)

            # Calculate the log_r
            log_r   = VAE_model.log_rz(sampled_z)
            log_p   = VAE_model.log_pz(sampled_z, mean, logvar)

            KLLoss = torch.mean(log_p - log_r, dim=0)

            # Accumulate the total loss 
            totalLoss = CELoss - beta * KLLoss

            avg_KL_loss += KLLoss.item()
            avg_CE_loss += CELoss.item()
            avg_epoch_loss += totalLoss.item()
            
            # Call back propagation
            totalLoss.backward()

            # Step in optimizer 
            optimizer.step()
        
        # Average the KL, CE, epoch loss
        avg_KL_loss /= len(TrainLoader)
        avg_CE_loss /= len(TrainLoader)
        avg_epoch_loss /= len(TrainLoader)

        # Obtain state population after every epoch
        newStates, state_population = VAE_model.get_Labels(TrainX)

        # calculate state population change
        state_population_change = torch.sqrt(torch.sum((state_population - state_population0)**2))

        # update state population 0 which is the reference
        state_population0 = state_population 

        # print if necessary
        if print_every!=-1 and (i+1) % print_every == 0:
            print("At Epoch {}".format(i+1))
            print("Average epoch Loss = {:.5f}".format(avg_epoch_loss))
            print("Average KL loss = {:.5f}".format(avg_KL_loss))
            print("Average CE loss = {:.5f}".format(avg_CE_loss))
            print("State population = ", state_population)
            print("State population change = {:.5f}".format(state_population_change))

        # Update labels and trainloader
        if update_labels and state_population_change < threshold:
            # Update the representative inputs
            VAE_model.update_representative_inputs(TrainX)

            Dataset = Loader(TrainX, newStates)
            TrainLoader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

            # reset the optimizer
            optimizer = optim.Adam(VAE_model.parameters(), lr=lr)

            print("Updating labels at epoch {}".format(i+1))
            print("State population = ", state_population)
            print("State population change = {:.5f}".format(state_population_change))

def PIB_train(Autoencoder_model, file_name:str, lr=1e-3):
    """
    """
    pass