import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    '''
    Define neural network class for enhancement module
    '''
    def __init__(self):
        super().__init__()
        modules_head = [nn.Conv2d(4, 32, 3, padding=(3//2))]
        self.head = nn.Sequential(*modules_head)
        
        modules_body = []
        for _ in range(4):
            modules_body.append(nn.Conv2d(32, 32, 3, padding=(3//2)))
            modules_body.append(nn.ReLU())
        modules_body.append(nn.Conv2d(32, 32, 3, padding=(3//2)))
        self.body = nn.Sequential(*modules_body)
        
        modules_tail = [
            nn.PixelShuffle(2),
            nn.Conv2d(8, 1, 3, padding=(3//2))]
        self.tail = nn.Sequential(*modules_tail)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        
    def forward(self, x):
        '''
        Define the forward pass: sequential + skip connections
        '''
        x_h = self.head(x)
        x_b = self.body(x_h) + x_h
        y   = self.tail(x_b) + self.upsample(x[:,0:2,:,:])
        return y

