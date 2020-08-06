import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_util import conv_dims, pool_dims

class Net1(nn.Module):
    
    def __init__(self, output_size, image_size=100, param_names=[], description=""):
        super(Net1, self).__init__()
        
        self.image_size = image_size
        self.output_size = output_size
        self.description = description
        self.param_names = param_names
        
        # first convolution
        i1, m1, n1 = self.image_size, 1, 8    # input size, num features in, num out
        k1, s1, p1 = 5, 2, 0              # kernel size, stride, padding
        o1 = conv_dims(i1, k1, s1, p1)    # output size
        # first pooling
        kp1, sp1 = 2, 2                   # kernel size, stride  
        op1 = pool_dims(o1, kp1, sp1)     # output size
        # second convolution
        i2, m2, n2 = op1, n1, 16          # input size, num features in, num out
        k2, s2, p2 = 3, 1, 0              # kernel size, stride, padding
        o2 = conv_dims(i2, k2, s2, p2)    # output size
        # second pooling
        kp2, sp2 = 2, 2                   # kernel size, stride
        op2 = pool_dims(o2, kp2, sp2)     # output size

        # fully connected layers
        self.mh0 = n2 * op2**2            # needed later - size of first linear layer
        mh1 = 1000                        # input size of second fully connected layer        
        mh2 = 50                          # input size of second fully connected layer        
        
        self.conv1 = nn.Conv2d(m1, n1, k1, stride=s1, padding=p1)
        self.conv2 = nn.Conv2d(m2, n2, k2, stride=s2, padding=p2)
        self.pool1 = nn.MaxPool2d(kp1, stride=sp1)
        self.pool2 = nn.MaxPool2d(kp2, stride=sp2)        

        self.fc1 = nn.Linear(self.mh0, mh1)
        self.fc2 = nn.Linear(mh1, mh2)
        self.fc3 = nn.Linear(mh2, self.output_size)
        
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.mh0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    def save_checkpoint(self, fpath):
        checkpoint = {
            "state_dict": self.state_dict(),
            "image_size": self.image_size,
            "output_size": self.output_size,
            "param_names": self.param_names,
            "description": self.description,
            # "history": epochs,
            # "lr": lr,
            #"optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, fpath)        
    
    
    @staticmethod
    def from_checkpoint(fpath):
        checkpoint = torch.load(fpath, map_location='cpu')
        print(checkpoint["description"])
        print("Input size: %g, output size: %g" % (checkpoint['image_size'], 
                                                   checkpoint['output_size']))
        print("parameter names: %s" % ", ".join(checkpoint['param_names']))

        model = Net1(checkpoint['output_size'], 
                     image_size=checkpoint['image_size'], 
                     description=checkpoint['image_size'], 
                     param_names=checkpoint['param_names'],
                    )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model




def train_regression(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=1):
    """Train a model. After training, the model weights are set to those that resulted 
    in the lowest validation loss during training.
    
    Parameters:
    - model: a model instance
    - dataloaders: a dictionary. Must have keys "train" and "val", and dataloaders as values
    - criterion: a loss function
    - optimizer: an optimizer instance
    - scheduler: a method from torch.optim.lr_scheduler
    - num_epochs: number of epochs to train (default 1)
    
    Returns:
    - history: dict with training/validation loss history
    """

    best_model_wts = copy.deepcopy(model.state_dict())

    # set initial loss to infinity
    best_loss = np.inf 

    dataset_sizes = {k: len(loader.dataset) for k, loader in dataloaders.items()}

    # initialise 
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}: '.format(epoch + 1, num_epochs), end="")
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data batches
            dataloader = dataloaders[phase]
            for batch_i, data in enumerate(dataloader):

                # get the input images and the shape parameters
                images, targets = data

                # flatten pts
                targets = targets.view(targets.size(0), -1)

                # convert variables to floats for regression loss
                #images = images.type(torch.FloatTensor)
                targets = targets.type(torch.FloatTensor)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()                
                
                # forward pass to get outputs
                output = model.forward(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output, targets)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics. Loss function returns mean loss over images; multiply by size of current batch
                running_loss += loss.item() * images.size(0)

            # learning rate scheduler
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # epoch finished
            epoch_loss = running_loss / dataset_sizes[phase]
            history[phase].append(epoch_loss)
            print('{} Loss: {:.6f} '.format(phase, epoch_loss), end="")
          
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # end of epoch
        print("\n")

    # final message
    print('\nBest val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return history
