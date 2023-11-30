from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np
import torch
from torch import nn, Tensor
from sklearn.metrics import r2_score

class R2Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction=reduction

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor=None) -> Tensor:
        ### pred: B, C, T, H, W. mask: H, W
        B = target.shape[0]
        if mask is None:
            mask = torch.ones_like(pred, device=pred.device)
        '''else:
            expanded_mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, H, W
            # Expand the mask tensor to match the shape of the input tensor
            # Using expand function for broadcasting
            mask = expanded_mask.expand(pred.size())
            mask = mask.to(pred.device)'''
        pred = pred*mask
        target = target*mask
        mask_flatten = mask.view(B, -1)
        target_flatten = target.view(B, -1)
        pred_flatten = pred.view(B, -1)
        # target_flatten_mean = torch.mean(target_flatten, dim=-1, keepdim=True)
        target_flatten_mean = torch.sum(target_flatten, dim=-1, keepdim=True)/torch.sum(mask_flatten, dim=-1, keepdim=True)
        rss = torch.sum(torch.square(target_flatten-pred_flatten), dim=-1) # B, 1
        tss = torch.sum(torch.square(target_flatten-target_flatten_mean), dim=-1) # B, 1
        r2_loss = rss/tss
        if self.reduction=='mean': # mean or sum over Batch. 
            result = torch.mean(r2_loss)
        elif self.reduction=='sum':
            result = torch.sum(r2_loss)
        return result
        # return F.mse_loss(input*mask, target*mask, reduction=self.reduction)

def trend(x):
    ### x: # batch, 1, time, lat, lon
    B, _, T, W, H = x.shape
    time_tensor = torch.arange(T, dtype=torch.float32).to(x.device)
    # Reshape the data tensor to (Batch, Time, -1), so each Lat-Lon grid point is a separate "feature"
    data_reshaped = x.view(B, T, -1)
    # Calculate the linear slopes for each sample in the batch
    mean_time = time_tensor.mean()
    time_diff = time_tensor - mean_time
    numerator = (time_diff.unsqueeze(0).unsqueeze(2) * data_reshaped).sum(dim=1)
    denominator = (time_diff**2).sum()
    slopes = numerator / denominator

    slopes = slopes.view(B, W, H)
    return slopes 


def generate_plot(x1, x2, x3, e): # x1: input, x2: target, x3: output, e: ensemble mean
    r2 = r2_score(x2.flatten(), x3.flatten())
    r2_base = r2_score(x2.flatten(), x1.flatten())
    xmin = np.min([np.min(x3), np.min(x2)])
    xmax = np.max([np.max(x3), np.max(x2)])
    xmax = np.max([np.abs(xmin), xmax])
    norm = colors.TwoSlopeNorm(vmin=-xmax, vcenter=0, vmax=xmax)
    xmin = np.min(x1)
    xmax = np.max(x1)
    xmax = np.max([np.abs(xmin), xmax])
    input_norm = colors.TwoSlopeNorm(vmin=-xmax, vcenter=0, vmax=xmax)
    xmin = np.min(e)
    xmax = np.max(e)
    xmax = np.max([np.abs(xmin), xmax])
    e_norm = colors.TwoSlopeNorm(vmin=-xmax, vcenter=0, vmax=xmax)
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(x1, origin='lower', norm=input_norm, cmap='coolwarm')
    ax.set_title('Input')
    plt.colorbar(mpl.cm.ScalarMappable(norm=input_norm, cmap='coolwarm'),
             ax=ax, shrink=.5)
    ax = fig.add_subplot(222)
    ax.imshow(x2, origin='lower', norm=norm, cmap='coolwarm')
    ax.set_title('Target')
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
             ax=ax, shrink=.5)
    xmin = np.min([np.min(x3), np.min(e)])
    xmax = np.max([np.max(x3), np.max(e)])
    xmax = np.max([np.abs(xmin), xmax])
    norm = colors.TwoSlopeNorm(vmin=-xmax, vcenter=0, vmax=xmax)
    ax = fig.add_subplot(223)
    ax.imshow(x3, origin='lower', norm=norm, cmap='coolwarm')
    ax.set_title('Output')
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'),
             ax=ax, shrink=.5)
    ax = fig.add_subplot(224)
    ax.imshow(e, origin='lower', norm=e_norm, cmap='coolwarm')
    ax.set_title('ENS')
    plt.colorbar(mpl.cm.ScalarMappable(norm=e_norm, cmap='coolwarm'),
             ax=ax, shrink=.5)
    plt.suptitle('r2: '+str(r2)+' base: '+str(r2_base))
    return fig
