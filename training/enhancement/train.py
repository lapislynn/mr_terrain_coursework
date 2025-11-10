import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from network import *

# Check if CUDA (GPU) is available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set a random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def augment_features(dem, grad_r=False, grad_c=False, grad_l2=False, grad_l1=False):
    '''
    Function to augment input features based on gradients
    '''
    if grad_r:
        gr = dem[:,0:1,:-1,:] - dem[:,0:1,1:,:]  # grad row
        gr = torch.cat((gr, gr[:,0:1,-1:,:]), axis=2)
        gr = gr * 10
        dem = torch.cat((dem, gr), axis=1)
    if grad_c:
        gc = dem[:,0:1,:,:-1] - dem[:,0:1,:,1:]  # grad col
        gc = torch.cat((gc, gc[:,0:1,:,-1:]), axis=3)
        gc = gc * 10
        dem = torch.cat((dem, gc), axis=1)
    if grad_l2:
        gr = dem[:,0:1,:-1,:] - dem[:,0:1,1:,:]  # grad row
        gr = torch.cat((gr, gr[:,0:1,-1:,:]), axis=2)
        gr = gr * 10
        gc = dem[:,0:1,:,:-1] - dem[:,0:1,:,1:]  # grad col
        gc = torch.cat((gc, gc[:,0:1,:,-1:]), axis=3)
        gc = gc * 10
        gl2 = torch.sqrt(torch.square(gr) + torch.square(gc))  # grad l2
        dem = torch.cat((dem, gl2), axis=1)
    if grad_l1:
        gr = dem[:,0:1,:-1,:] - dem[:,0:1,1:,:]  # grad row
        gr = torch.cat((gr, gr[:,0:1,-1:,:]), axis=2)
        gr = gr * 10
        gc = dem[:,0:1,:,:-1] - dem[:,0:1,:,1:]  # grad col
        gc = torch.cat((gc, gc[:,0:1,:,-1:]), axis=3)
        gc = gc * 10
        gl1 = torch.abs(gr) + torch.abs(gc)  # grad l1
        dem = torch.cat((dem, gl1), axis=1)
    return dem


# Initialize DataLoader instance for loading and processing data
dl = DataLoader(device=device)

# Learning rate and optimization setup for the model
LR = 0.001
model = [Net().to(device) for _ in range(N_LEVEL)]
optimizer = [optim.Adam(model[i].parameters(), lr=LR) for i in range(N_LEVEL)]
scheduler = [optim.lr_scheduler.StepLR(optimizer[i], step_size=2, gamma=0.8) for i in range(N_LEVEL)]

# Define loss functions for validation
criterion_valid = nn.MSELoss()

def criterion(op, gt):
    '''
    Loss function for training
    '''
    L1 = nn.L1Loss()
    loss = L1(op[:,0:1], gt[:,0:1])
    return loss

# Define training and validation parameters
N_EPOCH = 128
TRAIN_DEM_PER_EPOCH = len(dl.train_files)
TRAIN_ITER_PER_DEM = 64
TRAIN_BATCH_SIZE = 32
VALID_DEM_PER_EPOCH = 16
VALID_ITER_PER_DEM = 32
VALID_BATCH_SIZE = 32
TEACHER_FORCING_EPOCH = 0
assert TEACHER_FORCING_EPOCH <= N_EPOCH
LOSS_WEIGHTS = np.array([1.0,1.0,1.0,1.0,1.0])
LOSS_WEIGHTS /= np.sum(LOSS_WEIGHTS)
valid_loss_min = float('inf')


# Main training loop over epochs
for epoch in range(N_EPOCH):
    
    # Set models to training mode for gradient calculation
    for level in range(N_LEVEL):
        model[level].train()
    
    # Training phase
    loss_log = [0.0 for _ in range(N_LEVEL+1)]
    for i_dem in range(TRAIN_DEM_PER_EPOCH):
        dl.load_dem('train')
        for i_train_sample in range(TRAIN_ITER_PER_DEM):
            loss = torch.tensor(0.0).to(device)
            for level in range(N_LEVEL):
                if level == 0:
                    ip, gts, paths = dl.sample_path_batch(TRAIN_BATCH_SIZE)
                else:
                    if epoch < TEACHER_FORCING_EPOCH:
                        ip = nn.functional.grid_sample(gts[level-1], paths[level-1], mode='nearest', align_corners=True)
                    else:
                        ip = nn.functional.grid_sample(op[:,0:1],    paths[level-1], mode='nearest', align_corners=True)
                    ip = ip - ip.mean((2,3), True)
                ip = augment_features(ip, grad_l1=True, grad_r=True, grad_c=True)
                optimizer[level].zero_grad()
                op = model[level](ip)
                loss_m = criterion(op, augment_features(gts[level]))
                loss += LOSS_WEIGHTS[level] * loss_m
                loss_log[level] += float(loss_m)
            loss_log[-1] += float(loss)
            loss.backward()
            for level in range(N_LEVEL):
                nn.utils.clip_grad_value_(model[level].parameters(), 0.1)
                optimizer[level].step()
        
    for i in range(N_LEVEL+1):
        loss_log[i] /= (TRAIN_DEM_PER_EPOCH*TRAIN_ITER_PER_DEM)
            
    # Validation phase
    dl.reset_dem_idx('valid')
    for level in range(N_LEVEL):
        model[level].eval()
    valid_loss_epoch = 0.0
    
    with torch.no_grad():
        for i_dem in range(VALID_DEM_PER_EPOCH):
            dl.load_dem('valid')
            loss_log = [0.0 for _ in range(N_LEVEL+1)]
            for i_valid_sample in range(VALID_ITER_PER_DEM):
                loss = torch.tensor(0.0).to(device)
                for level in range(N_LEVEL):
                    if level == 0:
                        ip, gts, paths = dl.sample_path_batch(VALID_BATCH_SIZE)
                    else:
                        ip = nn.functional.grid_sample(op[:,0:1], paths[level-1], mode='nearest', align_corners=True)
                        ip = ip - ip.mean((2,3), True)
                    ip = augment_features(ip, grad_l1=True, grad_r=True, grad_c=True)
                    op = model[level](ip)[:,0:1,:,:]
                    loss_m = criterion_valid(op, gts[level])
                    loss += loss_m
                    loss_log[level] += float(loss_m)
                loss_log[-1] += float(loss)
                                
            for i in range(N_LEVEL+1):
                loss_log[i] /= VALID_ITER_PER_DEM
            valid_loss_epoch += loss_log[-1]
            
    # Adjust learning rates using the scheduler
    for level in range(N_LEVEL):
        scheduler[level].step()

    # Save models if validation loss improves
    if valid_loss_epoch < valid_loss_min:
        print("Validation Loss decreased %.6f -> %0.6f" % (valid_loss_min, valid_loss_epoch))
        valid_loss_min = valid_loss_epoch
        os.makedirs('weights/', exist_ok=True)
        for level in range(N_LEVEL):
            torch.save(model[level].state_dict(), 'weights/model_'+str(level)+'.pt')

