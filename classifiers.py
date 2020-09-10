import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from cnn_util import conv_dims, pool_dims


class ClfNet(nn.Module):
    
    def __init__(self, output_size, image_size=224, param_names=[], description=""):
        super(ClfNet, self).__init__()
        
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
        mh2 = 100                          # input size of second fully connected layer        
        
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
        x = F.log_softmax(self.fc3(x), dim=1)
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

        model = ClfNet(checkpoint['output_size'], 
                     checkpoint['image_size'], 
                     description=checkpoint['description'], 
                     param_names=checkpoint['param_names'],
                    )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model



class ClfNetBN(ClfNet):