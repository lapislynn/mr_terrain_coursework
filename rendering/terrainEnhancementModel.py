import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    '''
    Define a neural network class for terrain enhancement
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
        x_h = self.head(x)
        x_b = self.body(x_h) + x_h
        u = self.upsample(x[:,0:2,:,:])
        u[:,:,1:-1,1:-1] += self.tail(x_b)[:,:,1:-1,1:-1]
        return u


class TerrainEnhancementModel:
    '''
    Define a class for Terrain Enhancement Model
    '''
    def __init__(self, path='path/to/model'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        # try run
        self.predict(np.zeros((128,128)))

    def predict(self, ip):
        '''
        Perform prediction using the trained model
        '''
        ip_mean = ip.mean()
        ip = ip - ip_mean
        ip = ip[None,None,:,:]
        ip = self.augment_features(ip, grad_l1=True, grad_r=True, grad_c=True)
        
        with torch.no_grad():
            ip = torch.Tensor(ip).to(self.device)
            op = self.model(ip).cpu().numpy()[0,0]
        
        op = op + ip_mean
        return op


    def augment_features(self, dem, grad_r=False, grad_c=False, grad_l2=False, grad_l1=False):
        '''
        Augment input features with gradient information
        '''
        if grad_r:
            gr = dem[:,0:1,:-1,:] - dem[:,0:1,1:,:]  # grad row
            gr = np.concatenate((gr, gr[:,0:1,-1:,:]), axis=2)
            gr = gr * 10
            dem = np.concatenate((dem, gr), axis=1)
        if grad_c:
            gc = dem[:,0:1,:,:-1] - dem[:,0:1,:,1:]  # grad col
            gc = np.concatenate((gc, gc[:,0:1,:,-1:]), axis=3)
            gc = gc * 10
            dem = np.concatenate((dem, gc), axis=1)
        if grad_l1:
            gr = dem[:,0:1,:-1,:] - dem[:,0:1,1:,:]  # grad row
            gr = np.concatenate((gr, gr[:,0:1,-1:,:]), axis=2)
            gr = gr * 10
            gc = dem[:,0:1,:,:-1] - dem[:,0:1,:,1:]  # grad col
            gc = np.concatenate((gc, gc[:,0:1,:,-1:]), axis=3)
            gc = gc * 10
            gl1 = np.abs(gr) + np.abs(gc)  # grad l1
            dem = np.concatenate((dem, gl1), axis=1)
        return dem

