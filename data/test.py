from dataset import AnomalyDatasetN2N, YearDataset
from torch.utils.data import DataLoader

ds = YearDataset(model='CESM2', variable='tas', years=30, ds3d=True, signal=False)
print(ds.__len__())

loader = DataLoader(ds, batch_size=16, shuffle=True)
x, y, e = next(iter(loader))
print(x.shape)
print(y.shape)
print(e.shape)
print(x.min())
print(e.min())
print(x.max())
print(e.max())
