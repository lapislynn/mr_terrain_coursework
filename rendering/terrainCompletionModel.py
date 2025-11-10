import numpy as np
import torch
import torch.nn as nn
import torchvision


class Block(nn.Module):
    '''
    Define a basic convolutional block for the UNet architecture
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))


class Encoder(nn.Module):
    '''
    Define the encoder part of the UNet architecture
    '''
    def __init__(self, chs=(4,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    '''
    Define the decoder part of the UNet architecture
    '''
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs        = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    '''
    Define the complete UNet architecture
    '''
    def __init__(self, enc_chs=(4,64,128,256,512,1024), dec_chs=(1024,512,256,128,64), num_class=1):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out = torchvision.transforms.CenterCrop([128, 128])(out)
        return out


class TerrainCompletionModel:
    '''
    Define a model for terrain completion
    '''
    def __init__(self, path='weights/newest_gen.pt', gt_size=128):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.train()  # does not work if set to eval() :(
        self.gt_size = gt_size

        # Test the model with a sample input
        self.predict([None for i in range(9)])


    def predict(self, neighPatches):
        '''
        Predict the terrain completion for a central patch based on surrounding patches
        '''
        assert len(neighPatches) == 9, 'Error! Did not receive 9 patches as input. Send None if all not available.'
        assert neighPatches[4] is None, 'Error! Cannot have value at index 4. Index 4 will be predicted.'

        patch_mean = 0.0
        patch_std = 0.0
        patch_count = 0
        for p in neighPatches:
            if p is not None:
                patch_count += 1
                patch_mean += p.mean()
                patch_std += p.std()
        if patch_count > 0:
            patch_mean /= patch_count
            patch_std /= patch_count
        
        ip = np.zeros((1, 4, 3*self.gt_size, 3*self.gt_size))
        ip[0,1,self.gt_size:2*self.gt_size, self.gt_size:2*self.gt_size] = 1.0
        for i in range(3):
            for j in range(3):
                if i==1 and j==1:
                    continue
                elif neighPatches[i*3+j] is not None:
                    r, c = i*self.gt_size, j*self.gt_size
                    ip[0,2,r:r+self.gt_size, c:c+self.gt_size] = 1.0
                    ip[0,0,r:r+self.gt_size, c:c+self.gt_size] = neighPatches[i*3+j] - patch_mean
                else:
                    r, c = i*self.gt_size, j*self.gt_size
                    ip[0,3,r:r+self.gt_size, c:c+self.gt_size] = 1.0

        with torch.no_grad():
            ip = torch.Tensor(ip).to(self.device)
            op = self.model(ip).cpu().numpy()[0,0] + patch_mean

        return op
