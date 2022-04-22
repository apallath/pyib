### Training utility functions ###
import torch
from .models import VAE
from ..md.utils import TrajectoryReader
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np

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


def SPIB_data_prep(file_name:str, dt:int, label_file:str, one_hot=False, train_percent=0.8,comment_str="#", format="txyz", skip=1):
    """
    Function that prepares the data for SPIB training 

    Args:
        file_name(str)  : The file name of the trajectory file
        dt(int)         : The time lag (delta t)
        label_file(str) : The name of the label file, required to be a npy file
        one_hot(bool)   : Whether the labels are a one-hot encoding 
        train_percent(float)    : The percentage of data points to be used for training
    """
    # assert label file is an npy file
    assert label_file.endswith(".npy"), "We only take in .npy files for label file as of right now."

    traj_reader = TrajectoryReader(file_name, comment_char=comment_str, format=format)

    # read the trajectory
    time, traj = traj_reader.read_traj(skip=skip)
    traj = torch.tensor(traj, dtype=torch.float32)
    traj = (traj - traj.mean(axis=0))/traj.std(axis=0)
    traj = traj[:,:2]

    N = len(traj) - dt
    num_train = int(N * train_percent)

    # Select the training points
    randperm = torch.randperm(N)
    trainIdx  = randperm[:num_train]
    testIdx   = randperm[num_train:]

    # Create labels
    label = np.load(label_file)
    if one_hot:
        assert len(label.shape) == 2 , "If you choose the one hot encoding, the input data must be 2-dimensional while it is {}".format(len(label.shape))
        label = np.argmax(label, axis=1)
    label = torch.tensor(label).type(torch.LongTensor)


    # split train/test data/labels
    train_traj = traj[trainIdx,:2]
    train_labels = label[trainIdx + dt]
    test_traj  = traj[testIdx, :2]
    test_labels = label[testIdx + dt]

    return train_traj, train_labels, test_traj, test_labels, traj, label


def SPIB_train(VAE_model:VAE, filename:str, label_file:str, dt:int, lr=1e-3, update_labels=True, batch_size=1028, epochs=1000, \
    skip=1, beta=0.01, print_every=-1, threshold=0.01, patience=2, one_hot=False, device="cpu", comment_str="#", format="txyz"):
    """
    Function that performs the SPIB training process

    Args:
        VAE_model(torch.nn.Module)  : A torch model for beta-VAE
        filename(str)               : The filename of the trajectory input
        label_file(str)             : The filename of the label input
        dt(int)                     : The time lag used in the system
        lr(float)                   : The learning rate applied for the optimizer 
        update_labels(bool)         : Whether or not we update the labels
        batch_size(int)             : The batch size in which we perform training 
        epochs(int)                 : The number of epochs intended to be performed 
        skip(int)
        beta(float)                 : The beta present in the beta-VAE, the loss function is formulated as CE_Loss - beta * KL_Loss
        threshold(float)            : The threshold for the change in state population
        patience(int)               : Number of times to wait until we update the labels
    """
    # Define the losses 
    CrossEntropyLoss = nn.CrossEntropyLoss()

    # Prepare input data , data comes normalized
    TrainX, TrainY, testX, testY, traj, label = SPIB_data_prep(filename, dt, label_file, skip=skip, comment_str=comment_str, format=format, one_hot=one_hot)

    # Move data to device (could be GPU)
    TrainX = TrainX.to(device)
    TrainY = TrainY.to(device)
    testX  = testX.to(device)
    testY  = testY.to(device)
    VAE_model.to(device)

    # define optimizer 
    optimizer = optim.Adam(VAE_model.parameters(), lr=lr)

    # Define dataset, trainloader and testloader
    TrainDataset = Loader(TrainX, TrainY)
    TestDataset  = Loader(testX, testY)
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    TestLoader  = DataLoader(TestDataset, batch_size=batch_size)

    # before training, track the state population --> The first iteration is probably really bad 
    # I usually see it all predicting 1 state [ ...,0,0,0,1,0,0, ...]
    VAE_model.RepresentativeInputs_init(traj, label)
    _, state_population0 = VAE_model.get_Labels(TrainX)

    labels_update_count = 0

    for i in range(epochs):
        avg_epoch_loss = 0
        avg_KL_loss    = 0
        avg_CE_loss    = 0

        # set model to training mode
        VAE_model.train()

        for X, y in TrainLoader:
            # Clear the gradient in optimizer
            optimizer.zero_grad()

            # obtain the y prediction
            y_pred, mean, logvar, sampled_z = VAE_model(X)

            # Calculate the reconstruction loss or in this case crossentropyloss
            CELoss = CrossEntropyLoss(y_pred, y)

            # Calculate the log_r, log_p, KL = <log(p/r)>
            # Shape (N,1)
            log_r   = VAE_model.log_rz(sampled_z)
            log_p   = VAE_model.log_pz(sampled_z, mean, logvar)
            KLLoss = torch.mean(log_p - log_r, dim=0)

            # Accumulate the total loss 
            totalLoss = CELoss + beta * KLLoss
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

        # Perform testing 
        VAE_model.eval()
        avg_Loss_test = 0
        with torch.no_grad():
            for X , y in TestLoader:
                y_pred, mean, logvar, sampled_z = VAE_model(X)

                # Calculate the reconstruction loss or in this case crossentropyloss
                CELoss = CrossEntropyLoss(y_pred, y)

                # Calculate the log_r, log_p, KL = <log(p/r)>
                log_r   = VAE_model.log_rz(sampled_z)
                log_p   = VAE_model.log_pz(sampled_z, mean, logvar)
                KLLoss = torch.mean(log_p - log_r, dim=0)

                # totalLoss
                totalLoss = CELoss + beta * KLLoss

                avg_Loss_test += totalLoss.item()
            avg_Loss_test = avg_Loss_test / len(TestLoader)
            newTestStates, _ = VAE_model.get_Labels(testX)

        # print if necessary
        if print_every!=-1 and (i+1) % print_every == 0:
            print("-----------------------------------")
            print("At Epoch {}".format(i+1))
            print("Average epoch Loss = {:.5f}".format(avg_epoch_loss))
            print("Average KL loss = {:.5f}".format(avg_KL_loss))
            print("Average CE loss = {:.5f}".format(avg_CE_loss))
            print("Average test Loss = {:.5f}".format(avg_Loss_test))
            print("State population = ", state_population)
            print("State population change = {:.5f}".format(state_population_change))

        # Update labels and trainloader
        if update_labels and state_population_change < threshold:
            labels_update_count += 1
            if labels_update_count > patience:
                # Update the representative inputs
                VAE_model.update_representative_inputs(TrainX)

                # Update training set
                Dataset = Loader(TrainX, newStates)
                TrainLoader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

                # Update test set
                TestSet = Loader(testX, newTestStates)
                TestLoader = DataLoader(TestSet, batch_size=batch_size)

                # reset the optimizer
                optimizer = optim.Adam(VAE_model.parameters(), lr=lr)
                print("###################################")
                print("Updating labels at epoch {}".format(i+1))

                labels_update_count = 0