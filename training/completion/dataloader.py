import numpy as np
import rasterio
import glob
import cv2
import random
from einops import rearrange, pack, repeat
import torch
import torchvision


class DataLoader():
    def __init__(self, dataset_path='dataset', split=[0.8, 0.1, 0.1], device=torch.device('cpu')):
        '''
        Initializes a DataLoader object to manage loading and processing of DEM (Digital Elevation Model) data.
        '''
        self.device = device

        if split[0] + split[1] + split[2] != 1.0:
            print('Error: Ratio of split not equal to 1.0!')
        
        files = glob.glob(dataset_path + '/*')
        if len(files) == 0:
            print('Error: No files found!')
        elif files[0][-3:] != 'tif':
            print('Error: Non-tif file found in directory!')
                    
        N_files = len(files)
        N_train_files = round(N_files * split[0])
        N_valid_files = round(N_files * split[1])
        
        self.train_files = files[:N_train_files]
        self.valid_files = files[N_train_files:N_train_files+N_valid_files]
        self.test_files  = files[N_train_files+N_valid_files:]
        
        self.spo2 = 2**13  # size in power of 2 to crop
        self.dem = None
        
        self.train_file_idx = 0
        self.valid_file_idx = 0
        self.test_file_idx  = 0
        
        self.dem_size = 512
        self.gt_size = 128
        
        self.train_dem_list = []
        self.valid_dem_list = []
        self.test_dem_list  = []

        # _, self.std = self.compute_dataset_stats()
        '''
        Dataset Statistics:
        Global Mean: 790.6718
        Global Std:  506.7215
        Data Range:  [4.87, 2022.16]
        '''
        # self.mean = 790.6718
        self.std = 506.7215
        # self.std = 40

    def compute_dataset_stats(self):
        all_data = []

        all_files = self.train_files + self.valid_files + self.test_files

        for file in all_files:
            data = rasterio.open(file).read(1)[:self.spo2, :self.spo2]
            data = cv2.resize(data, (self.dem_size, self.dem_size), interpolation=cv2.INTER_CUBIC)
            all_data.append(data)

        all_data = np.array(all_data)

        global_mean = np.mean(all_data)
        global_std = np.std(all_data)

        print(f"Dataset Statistics:")
        print(f"Global Mean: {global_mean:.4f}")
        print(f"Global Std:  {global_std:.4f}")
        print(f"Data Range:  [{np.min(all_data):.2f}, {np.max(all_data):.2f}]")

        return global_mean, global_std
        
    def load_dem(self, file_type='train'):
        '''
        Loads a DEM file from the specified type of dataset split.
        '''

        if file_type == 'train' and self.train_file_idx < len(self.train_dem_list):
            self.dem = self.train_dem_list[self.train_file_idx]
            self.train_file_idx = (self.train_file_idx + 1) % len(self.train_files)
            return
        elif file_type == 'valid' and self.valid_file_idx < len(self.valid_dem_list):
            self.dem = self.valid_dem_list[self.valid_file_idx]
            self.valid_file_idx = (self.valid_file_idx + 1) % len(self.valid_files)
            return
        elif file_type == 'test' and self.test_file_idx < len(self.test_dem_list):
            self.dem = self.test_dem_list[self.test_file_idx]
            self.test_file_idx = (self.test_file_idx + 1) % len(self.test_files)
            return
        
        if file_type == 'train':
            file = self.train_files[self.train_file_idx]
            self.train_file_idx = (self.train_file_idx + 1) % len(self.train_files)
        elif file_type == 'valid':
            file = self.valid_files[self.valid_file_idx]
            self.valid_file_idx = (self.valid_file_idx + 1) % len(self.valid_files)
        elif file_type == 'test':
            file = self.test_files[self.test_file_idx]
            self.test_file_idx = (self.test_file_idx + 1) % len(self.test_files)
        else:
            print('Error: Invalid file type! Valid types are: train/valid/test.')
            return

        data = rasterio.open(file).read(1)[:self.spo2, :self.spo2]
        # data[data == -999999.0] = 0
        data = cv2.resize(data, (self.dem_size, self.dem_size), interpolation = cv2.INTER_CUBIC)
        data -= np.mean(data)
        # data -= self.mean
        data /= self.std  # 40.0 (==dataset std)
        self.dem = data
        
        if file_type == 'train':
            self.train_dem_list.append(self.dem)
        elif file_type == 'valid':
            self.valid_dem_list.append(self.dem)
        elif file_type == 'test':
            self.test_dem_list.append(self.dem)
            
    def reset_dem_idx(self, file_type='valid'):
        '''
        Resets the index for loading DEM files from the specified type of dataset split.
        '''
        if file_type == 'train':
            self.train_file_idx = 0
        elif file_type == 'valid':
            self.valid_file_idx = 0
        elif file_type == 'test':
            self.test_file_idx = 0
        else:
            print('Error: Invalid file type in reset! Valid types are: train/valid/test.')
    
    def sample(self):
        '''
        Sampling of a processed patch
        '''
        if self.dem is None:
            print('Error: DEM not loaded yet!')
            return
        
        if random.randint(0,1) == 1:
            self.dem = np.flip(self.dem, axis=1)
        self.dem = np.rot90(self.dem, random.randint(0,3))
        
        r, c = random.randint(0,self.dem_size-3*self.gt_size), random.randint(0,self.dem_size-3*self.gt_size)
        patch = self.dem[r:r+3*self.gt_size, c:c+3*self.gt_size]
        
        mask = np.zeros((3*self.gt_size,3*self.gt_size))
        for i in range(3):
            for j in range(3):
                if random.randint(0,1) == 1:
                    r, c = i*self.gt_size, j*self.gt_size
                    mask[r:r+self.gt_size, c:c+self.gt_size] = 1.0
        mask[self.gt_size:2*self.gt_size, self.gt_size:2*self.gt_size] = 1.0
        patch = patch * mask
        mask_absent = 1.0 - mask        
        mask[self.gt_size:2*self.gt_size, self.gt_size:2*self.gt_size] = 0.0
        
        mask_pred = np.zeros((3*self.gt_size,3*self.gt_size))
        mask_pred[self.gt_size:2*self.gt_size, self.gt_size:2*self.gt_size] = 1.0
        im, _ = pack([patch, mask_pred, mask, mask_absent], '* h w')
        return im
    
    def sample_batch(self, batch_size):
        '''
        Samples a batch of processed patches
        '''
        batch = [self.sample() for i in range(batch_size)]
        batch, _ = pack(batch, '* c h w')        
        batch = torch.Tensor(batch).to(self.device)
        return batch

