import xarray as xa
import numpy as np
from torch.utils.data import Dataset
import glob
import torch
from tqdm import tqdm

class YearDataset(Dataset):
    def __init__(self, model, variable, years=1, ds3d=True, signal=False, 
                 range01=False, random=True) -> None:
        super().__init__()
        path = '/p/gpfs1/shiduan/ForceSMIP/Training/YearAnomaly/' # Lassen
        # path = '/p/lustre1/shiduan/ForceSMIP/Training/YearAnomaly/' # Other than Lassen
        # files = glob.glob(path+model+'-'+variable+'-stdanomaly-r*.nc')
        files = glob.glob(path+model+'-'+variable+'-anomaly-r*.nc')
        self.input_data = []
        input_data = []
        self.n_ensembles = len(files) # how many ensemble members we have. 
        self.years = years
        self.ds3d = ds3d
        self.signal = signal
        self.random = random
        ens = []
        vmins = []
        vmaxs = []
        for file in tqdm(sorted(files)):
            data = xa.open_dataset(file)
            data = data[variable]
            vmin = data.min(dim='time')
            vmax = data.max(dim='time')
            vmins.append(np.expand_dims(vmin.data, axis=0))
            vmaxs.append(np.expand_dims(vmax.data, axis=0))
            t, x, y = data.shape
            input_data.append(torch.from_numpy(
                data.data).float())
            ens.append(data.data.reshape(1, t, x, y))
        self.time = t # how many months we have for one ensemble member. 
        ens = np.concatenate(ens, axis=0)
        self.ens = np.mean(ens, axis=0) # t, x, y
        self.ens = torch.from_numpy(self.ens).float()
        print(self.ens.shape)
        # 0-1 process
        if range01:
            vmins = np.concatenate(vmins, axis=0)
            vmaxs = np.concatenate(vmaxs, axis=0)
            vmins = np.min(vmins, axis=0)
            vmaxs = np.max(vmaxs, axis=0)
            print(vmins.shape, vmaxs.shape)
            self.ens = (self.ens-vmins)/(vmaxs.data-vmins)
            for x in input_data:
                x = (x-vmins)/(vmaxs-vmins)
                self.input_data.append(x.float())
        else:
            for x in input_data:
                self.input_data.append(x.float())
        print(np.min(self.input_data[0].data.numpy())) # should close to 0. 
        print(np.max(self.input_data[1].data.numpy())) # should close to 1. 
    
    def find_file(self, x, n):
        file_index = x // n
        time_step = x % n
        return file_index, time_step
    def __len__(self):
        length = self.n_ensembles*self.time
        length = self.n_ensembles*(self.time-self.years-1)
        return length
    
    def __getitem__(self, index):
        if not self.signal:
            if self.random:
                index1 = np.random.randint(self.n_ensembles)
                index2 = np.random.randint(self.n_ensembles)
            else:
                index1 = 1
                index2 = 2
            while index1==index2:
                index2 = np.random.randint(self.n_ensembles)
            if self.random:
                time_step = np.random.randint(self.time-self.years-1)
            else: # fix visualization. 
                time_step = 10
            x = self.input_data[index1][time_step:time_step+self.years]
            y = self.input_data[index2][time_step:time_step+self.years] # time, lat, lon
            e = self.ens[time_step:time_step+self.years] # time, lat, lon
        else:
            file_index, time_step = self.find_file(index, self.time-self.years-1)
            x = self.input_data[file_index][time_step:time_step+self.years]
            y = self.input_data[file_index][time_step:time_step+self.years] # time, lat, lon
            e = self.ens[time_step:time_step+self.years] # time, lat, lon
        if self.ds3d:
            x = x.unsqueeze(0) # 1, time, lat, lon
            y = y.unsqueeze(0)
            e = e.unsqueeze(0)
        
        return x, y, e