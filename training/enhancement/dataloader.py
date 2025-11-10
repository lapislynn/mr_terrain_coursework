import numpy as np
import rasterio
import glob
import cv2
import random
import torch


# Number of enhancement models for progressive super-resolution
N_LEVEL = 5


class DataLoader():
    def __init__(self, dataset_path='path/to/dataset', split=[0.8, 0.1, 0.1], device=torch.device('cpu')):
        '''
        Initializes a DataLoader object to manage loading and processing of DEM data.
        '''
        self.device=device

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
        self.dem_mean = None
        
        self.train_file_idx = 0
        self.valid_file_idx = 0
        self.test_file_idx  = 0
        
        self.gt_size = 256
        ls = np.linspace(-1,1,self.gt_size, dtype=np.float32)
        xx = np.tile(ls, (self.gt_size,1))
        yy = xx.T
        self.grid = np.concatenate((xx[:,:,None],yy[:,:,None]), axis=2)
        
        
    def load_dem(self, file_type='train'):
        '''
        Loads a DEM file from the specified type of dataset split.
        '''
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
        
        self.dem_mean = np.mean(data)
        data -= self.dem_mean
        data /= 40.0
        self.dem = []
        for i in range(N_LEVEL,-1,-1):  # range(N_LEVEL+1)
            size = self.spo2 // 2**i
            self.dem.append(cv2.resize(data, (size, size), interpolation = cv2.INTER_CUBIC))
    
    
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
    
    
    def sample_path(self):
        '''
        Quad-tree based sampling of a processed patch
        '''
        if self.dem is None:
            print('Error: DEM not loaded yet!')
            return
        
        path = [random.randint(1,4) for _ in range(N_LEVEL)]
        
        def give_patch(path_len, row, col, size, idx):
            '''
            Resursively get tiles of finer resolution
            '''
            if idx == path_len:
                return (row, col, size)
            else:
                hsize = size // 2
                if   path[idx] == 1:
                    return give_patch(path_len, row      , col+hsize, hsize, idx+1)
                elif path[idx] == 2:
                    return give_patch(path_len, row      , col      , hsize, idx+1)
                elif path[idx] == 3:
                    return give_patch(path_len, row+hsize, col      , hsize, idx+1)
                elif path[idx] == 4:
                    return give_patch(path_len, row+hsize, col+hsize, hsize, idx+1)
                else:
                    print('Error: invalid path! Path should lie in [1,4].')
                    return
                    
        (row, col, size) = give_patch(1, 0, 0, self.dem[0].shape[0], 0)
        ip = self.dem[0][row:row+size,col:col+size]
        ip -= ip.mean()
        
        gts = []
        for i in range(1,N_LEVEL+1):
            (row, col, size) = give_patch(i, 0, 0, self.dem[i].shape[0], 0)
            gt = self.dem[i][row:row+size,col:col+size]
            gt -= gt.mean()
            gts.append(gt)
            
        return ip, gts, path[1:]
    
    
    def sample_path_batch(self, batch_size):
        '''
        Samples a batch of processed patches
        '''
        ip_list   = []
        gts_list  = [[] for _ in range(N_LEVEL)]
        path_list = [[] for _ in range(N_LEVEL-1)]
        
        def process_path(p):
            hgt_size = self.gt_size // 2
            if   p == 1:
                return self.grid[:hgt_size, hgt_size:]
            elif p == 2:
                return self.grid[:hgt_size, :hgt_size]
            elif p == 3:
                return self.grid[hgt_size:, :hgt_size]
            elif p == 4:
                return self.grid[hgt_size:, hgt_size:]
            else:
                print('Error: invalid path in batch!')
        
        for batch in range(batch_size):
            ip, gts, path = self.sample_path()
            
            ip_list.append(ip[None,None])
            
            for i in range(N_LEVEL):
                gts_list[i].append(gts[i][None,None])
            
            for i in range(N_LEVEL-1):
                path_list[i].append(process_path(path[i])[None])
                
        b_ip    = torch.Tensor(np.concatenate(ip_list, axis=0)).to(self.device)
        b_gts   = [torch.Tensor(np.concatenate(g, axis=0)).to(self.device) for g in gts_list]
        b_paths = [torch.Tensor(np.concatenate(p, axis=0)).to(self.device) for p in path_list]
        
        return b_ip, b_gts, b_paths
