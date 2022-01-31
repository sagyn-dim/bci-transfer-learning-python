#Creating custom dataset
#Import the libraries
import torch
import pickle
# import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from scipy import signal
import math

#Normalization values
mx = 2.8495165110816303e-05
mn = 0
slope = 2/(mx-mn)

#Custom dataset 
#The custom dataset has to extend the Dataset class
class cust_dataset(Dataset):
    
    data_raw = [] #Need the attribute to store the list of obj
    
    #The input is the name of a dataset, has to include .pickle extension
    def __init__(self, data_raw, transform=None):
        """
       
        """
        self.data_raw = data_raw
        self.transform = transform

    def __len__(self):
        sample_len = 0
        for sub in self.data_raw:
            sample_len += len(sub)
        return sample_len

    def __getitem__(self, idx):
        #The function will expect to return only one sample at a time
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Only the requested sample will be located and returned
        #Locate
        img = 0
        label = 0
        track_idx = 0
        #Iterate through subjects
        for ss in range(len(self.data_raw)): 
            if (track_idx + len(self.data_raw[ss])) > idx: 
                #Iterate through events
                for ii in range(len(self.data_raw[ss])):
                    #Store the sample in img
                    if (track_idx + ii) == idx: 
#                         print('In')
                        epoch_obj = self.data_raw[ss][ii]
                        img = torch.from_numpy(epoch_obj.get_data())
                        try:
                            label = epoch_obj.event_id['left_hand'] - 1
                        except:
                            label = 1
                        break
                if torch.is_tensor(img): break
                    
            else: 
                track_idx = track_idx + len(self.data_raw[ss])
        
#         if self.transform:
#             sample = self.transform(sample)

        return img, label

#Now let's create a function which will: 
#return training_ds and valid_ds in dictionary
#and return training_dl and valid_dl in dictionary
#Taking in a whole unextended dataset
def gener_ds (file_name,batch_size,num_workers, ss_num):
    with open(file_name, 'rb') as handle:
        data_raw = pickle.load(handle)
    #Test run, use only first two subjects
    # data_raw = data_raw[:2]
    #Leave out one subject for validation, e.g the last one
    val_sub = []
    val_sub.append(data_raw.pop(ss_num))
    train_subs = data_raw
    #Use the custom dataset class
    train_ds = cust_dataset(train_subs) 
    val_ds = cust_dataset(val_sub)
    # Datasets
    datasets_dict = {'train': train_ds, 'val': val_ds}
    # Dataloaders
    dataloaders_dict = {x: DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']} 
    return datasets_dict, dataloaders_dict
    
#The function applies stft and normalizes the data batch (transformed range: [-1 1])
def stft_normalize(images_dl, grid_h = 16, grid_len = 18):
    stft_out = np.zeros([len(images_dl), 3, 224, 224])
    #iterate through events
    for ee in range(len(images_dl)):
        #iterate through channels
        pointer = 0
        for cc in range(images_dl.shape[2]):
            workArr = images_dl[ee,0,cc,:]
            f, t, Zxx = signal.stft(workArr, fs = 250, nperseg=35, noverlap = 13) #Stft applied
            Zxx = abs(Zxx)
            for yy in range(7):
                row = math.floor(pointer/14) * grid_len
                column = (pointer%14) * grid_h
                pointer = pointer + 1
                for ww in range(3):
                    stft_out[ee, ww, row:row+grid_len, column:column+grid_h] = (Zxx - mn)*slope - 1
    images_dl = 0
    return torch.from_numpy(stft_out)