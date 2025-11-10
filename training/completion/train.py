import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle
from dataloader import *
from network import *
import matplotlib.pyplot as plt

def preview(fake, real, epoch, batch):
    fake = fake.detach().cpu().numpy()
    real = real.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].imshow(fake, cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].tick_params(axis='both', which='both', length=0)
    ax[0].set_title('Fake')

    ax[1].imshow(real, cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].tick_params(axis='both', which='both', length=0)
    ax[1].set_title('Real')

    plt.savefig(f'outputs/{epoch:03d}-{batch:03d}.png', bbox_inches='tight', pad_inches=0.2, format='png')


def pretrain_G(epochs=10):
    print('Pre-training generator G.')
    net_g.train()
    optimizer_pretrain = optim.Adam(net_g.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion_pretrain = nn.L1Loss()

    for epoch in range(epochs):
        dl.reset_dem_idx('train')
        total_loss = 0
        batch_count = 0

        print(f'Epoch {epoch+1:03d}/{epochs:03d}')
        for i_train_file in range(len(dl.train_files)):
            dl.load_dem('train')
            real = dl.sample_batch(BATCH_SIZE)

            ip_g = real[:, :, :, :].clone()
            ip_g[:, 0, 128:-128, 128:-128] = 0.0  # Mask center region

            net_g.zero_grad()
            op_g = net_g(ip_g)

            loss_pretrain = criterion_pretrain(op_g, real[:, 0:1, 128:-128, 128:-128])
            print(f'\tBatch {batch_count+1:03d}, loss_g: {loss_pretrain:.4f}')
            loss_pretrain.backward()
            optimizer_pretrain.step()

            total_loss += loss_pretrain.item()
            batch_count += 1

    print("Generator pre-training completed!")
    torch.save(net_g.state_dict(), 'weights/pretrained_gen.pt')
    return net_g


def get_soft_labels(batch_size, device, real_label=True, add_noise=True):
    if real_label:
        base_labels = torch.ones((batch_size)).to(device)
        if add_noise:
            noise = torch.FloatTensor(batch_size).uniform_(-0.2, 0).to(device)
            return base_labels + noise
    else:
        base_labels = torch.zeros((batch_size)).to(device)
        if add_noise:
            noise = torch.FloatTensor(batch_size).uniform_(0, 0.2).to(device)
            return base_labels + noise

    return base_labels


# Check and set the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set a random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize the data loader
dl = DataLoader(device=device)

# Hyperparameters for training
LR = 0.002
BATCH_SIZE = 8
N_EPOCH = 256
# N_EPOCH = 8
LAMBDA_L1 = 10
D_UPDATE_INTERVAL = 1

# Initialize generator and discriminator networks and move them to the selected device
net_g = UNet().to(device)
net_d = NetD().to(device)

# # Pre-train G
# net_g = pretrain_G(epochs=10)

# Initialize optimizers for generator and discriminator networks
optimizer_g = optim.Adam(net_g.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=LR, betas=(0.5, 0.999))

scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=4, gamma=0.75)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=4, gamma=0.75)

# Define loss functions for training
criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# Create a directory to save weights if it doesn't exist
os.makedirs('weights/', exist_ok=True)

labels_real = torch.ones((BATCH_SIZE)).to(device)
labels_fake = torch.zeros((BATCH_SIZE)).to(device)

loss_history = {
    'd_real': [],
    'd_fake': [],
    'g_bce': [],
    'g_l1': [],
    'g_total': []
}
best_g_loss = float('inf')
best_epoch = -1
d_update_counter = 0

# Loop through epochs for training
for epoch in range(N_EPOCH):
    dl.reset_dem_idx('train')
    net_g.train()
    net_d.train()

    epoch_losses = {
        'd_real': [],
        'd_fake': [],
        'g_bce': [],
        'g_l1': [],
        'g_total': []
    }
    batch_count = 0

    print(f'Epoch {epoch + 1:03d}/{N_EPOCH:03d}')

    # Loop through training data
    for i_train_file in range(len(dl.train_files)):
        dl.load_dem('train')
        batch_count += 1

        update_d = (d_update_counter % D_UPDATE_INTERVAL == 0)
        if update_d:
            soft_labels_real = get_soft_labels(BATCH_SIZE, device, real_label=True, add_noise=True)
            soft_labels_fake = get_soft_labels(BATCH_SIZE, device, real_label=False, add_noise=True)

            # Train discriminator on real data
            net_d.zero_grad()
            real = dl.sample_batch(BATCH_SIZE)
            # op_d_real = net_d(real[:, :, 96:-96, 96:-96]).view(-1)
            op_d_real = net_d(real[:, [0], 96:-96, 96:-96]).view(-1)
            # loss_d_real = criterion_bce(op_d_real, soft_labels_real)
            loss_d_real = criterion_bce(op_d_real, labels_real)
            loss_d_real.backward()
            epoch_losses['d_real'].append(loss_d_real.item())
            print(f'\tBatch {batch_count:03d}, loss_d_real: {loss_d_real:.4f}')

            # Train discriminator on fake data generated by the generator
            ip_g = real[:, :, :, :].clone()
            ip_g[:, 0, 128:-128, 128:-128] = 0.0
            op_g_d = net_g(ip_g)

            fake = nn.functional.pad(op_g_d, (128, 128, 128, 128), "constant", 0)
            fake = fake + (1.0 - real[:, 1:2, :, :].clone()) * real[:, 0:1, :, :].clone()
            fake = torch.cat((fake, real[:, 1:, :, :].clone()), dim=1)
            # op_d_fake = net_d(fake[:, :, 96:-96, 96:-96].detach()).view(-1)
            op_d_fake = net_d(fake[:, [0], 96:-96, 96:-96].detach()).view(-1)
            # loss_d_fake = criterion_bce(op_d_fake, soft_labels_fake)
            loss_d_fake = criterion_bce(op_d_fake, labels_fake)
            loss_d_fake.backward()
            optimizer_d.step()
            epoch_losses['d_fake'].append(loss_d_fake.item())
            print(f'\tBatch {batch_count:03d}, loss_d_fake: {loss_d_fake:.4f}')
        else:
            real = dl.sample_batch(BATCH_SIZE)
            print(f'\tBatch {batch_count:03d}, [D skipped]')

        # Train generator
        net_g.zero_grad()
        if not update_d:
            ip_g = real[:, :, :, :].clone()
            ip_g[:, 0, 128:-128, 128:-128] = 0.0
            op_g_d = net_g(ip_g)
            fake = nn.functional.pad(op_g_d, (128, 128, 128, 128), "constant", 0)
            fake = fake + (1.0 - real[:, 1:2, :, :].clone()) * real[:, 0:1, :, :].clone()
            fake = torch.cat((fake, real[:, 1:, :, :].clone()), dim=1)

        soft_labels_real_gen = get_soft_labels(BATCH_SIZE, device, real_label=True, add_noise=False)
        # op_g = net_d(fake[:, :, 96:-96, 96:-96]).view(-1)
        op_g = net_d(fake[:, [0], 96:-96, 96:-96]).view(-1)
        # loss_g_bce = criterion_bce(op_g, soft_labels_real_gen)
        loss_g_bce = criterion_bce(op_g, labels_real)
        loss_g_l1 = criterion_l1(fake[:, 0, 128:-128, 128:-128], real[:, 0, 128:-128, 128:-128])
        # w1 = loss_g_l1.item() / (loss_g_bce.item() + loss_g_l1.item())
        # w2 = loss_g_bce.item() / (loss_g_bce.item() + loss_g_l1.item())
        # loss_g = w1 * loss_g_bce + w2 * loss_g_l1
        loss_g = loss_g_bce + LAMBDA_L1 * loss_g_l1
        loss_g.backward()
        optimizer_g.step()
        if batch_count % 4 == 0:
            preview(fake[0, 0], real[0, 0], epoch+1, batch_count)
        epoch_losses['g_bce'].append(loss_g_bce.item())
        epoch_losses['g_l1'].append(loss_g_l1.item())
        epoch_losses['g_total'].append(loss_g.item())
        print(f'\tBatch {batch_count:03d}, loss_g: {loss_g:.4f}, loss_g_bce: {loss_g_bce:.4f}, loss_g_l1: {loss_g_l1:.4f}')

        d_update_counter += 1

    for key in epoch_losses:
        loss = np.array(epoch_losses[key]).mean().item()
        loss_history[key].append(loss)

    current_g_loss = loss_history['g_total'][-1]
    if current_g_loss < best_g_loss:
        best_g_loss = current_g_loss
        best_epoch = epoch
        torch.save(net_g.state_dict(), f'weights/best_gen_{epoch + 1:03d}.pt')
        torch.save(net_d.state_dict(), f'weights/best_dis_{epoch + 1:03d}.pt')
        print(f'Epoch {epoch + 1}: New best model saved with G_loss: {current_g_loss:.4f}')

    # # Save generator and discriminator weights for this epoch
    # torch.save(net_g.state_dict(), 'weights/gen_'+str(epoch)+'.pt')
    # torch.save(net_d.state_dict(), 'weights/dis_'+str(epoch)+'.pt')

    # Save newest generator and discriminator weights
    torch.save(net_g.state_dict(), 'weights/newest_gen.pt')
    torch.save(net_d.state_dict(), 'weights/newest_dis.pt')

    # Adjust learning rates using schedulers
    scheduler_g.step()
    scheduler_d.step()

print(f"Training completed! Best model from epoch {best_epoch + 1} with G_loss: {best_g_loss:.4f}")

with open(f"weights/loss.pkl", "wb") as f:
    pickle.dump(loss_history, f)
