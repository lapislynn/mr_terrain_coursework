import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class Block(nn.Module):
    '''
    Defines a basic building block for the UNet architecture
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
    Defines the encoder part of the UNet architecture
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
    Defines the decoder part of the UNet architecture
    '''
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
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
    Defines the complete UNet architecture
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


class NetD(nn.Module):
    '''
    Defines a discriminator network
    '''
    def __init__(self, ip_channels=1, body_channels=64, tail_channels=1, body_size=3):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(ip_channels, body_channels, 3),
            nn.LeakyReLU(0.2)
        )
        
        conv_body_list = []
        for i in range(body_size):
            conv_body_list.append(nn.Conv2d(body_channels,body_channels, 3))
            conv_body_list.append(nn.BatchNorm2d(body_channels))
            conv_body_list.append(nn.LeakyReLU(0.2))
            conv_body_list.append(nn.Conv2d(body_channels,body_channels, 3))
            conv_body_list.append(nn.MaxPool2d(2))
        self.conv_body = nn.Sequential(*conv_body_list)
        
        self.conv_tail = nn.Conv2d(body_channels,tail_channels, 3)
        
        self.classifier = nn.Sequential(
            nn.Linear(324, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv_body(x)
        x = self.conv_tail(x)
        x = torch.flatten(x, start_dim=1)
        y = self.classifier(x)
        return y
