import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from .utils import generate_plot
import pytorch_lightning as pl

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        # out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    def __init__(self, blocks, in_channels, hidden_channels, out_channels, kernel_size):
        super(ResNet3D, self).__init__()
        print(in_channels, hidden_channels)
        self.layer1 = self._spatial_consistent_block(in_channels=in_channels, out_channels=hidden_channels,
                                                     kernel_size=kernel_size, relu=True)
        # self.maxpool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.layer2 = self._spatial_consistent_block(in_channels=hidden_channels, out_channels=hidden_channels,
                                                     kernel_size=kernel_size, relu=True)
        # self.maxpool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.res_block = nn.Sequential(*[BasicBlock3D(in_channels=hidden_channels, 
                                       out_channels=hidden_channels, kernel_size=kernel_size) 
                                       for _ in range(blocks)])
        self.up1 = self._upsample(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.up2 = self._upsample(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.out_conv = self._spatial_consistent_block(in_channels=hidden_channels, out_channels=out_channels, 
                                                       kernel_size=kernel_size, relu=False)
    
    def _spatial_consistent_block(self, in_channels, out_channels, kernel_size, relu=True):
        conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                          stride=1, padding=kernel_size//2, padding_mode='reflect')
        bn1 = nn.BatchNorm3d(out_channels)
        if relu:
            relu = nn.ReLU(inplace=True)
            return nn.Sequential(conv1, bn1, relu)
        else:
            return conv1
        
    def _upsample(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding_mode='zeros'),
            # Only "zeros" padding mode is supported for ConvTranspose3d
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels)
            # nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.res_block(x)
        x = self.up2(self.up1(x))
        # print(x.shape, ' before out_conv')
        x = self.out_conv(x)
        # print(x.shape, 'after out_conv')

        return x

class ResNet3DLightning(pl.LightningModule):
    def __init__(self, blocks, in_channels, out_channels, hidden_channels, kernel_size, lr=1e-3, 
                 signal=False, unforced=False):
        super(ResNet3DLightning, self).__init__()
        self.save_hyperparameters()
        self.model = ResNet3D(blocks, in_channels, hidden_channels, out_channels, kernel_size)
        if signal and unforced:
            self.model = ResNet3D(blocks, in_channels, hidden_channels, out_channels*2, kernel_size)
        self.lr = lr
        self.signal = signal
        self.unforced = unforced
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
        # return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    

    def training_step(self, batch, batch_idx):
        x, y, e = batch
        y_pred = self(x)
        if self.signal and (not self.unforced):
            loss = nn.functional.mse_loss(y_pred, e)
            # loss = nn.functional.l1_loss(y_pred, e)
        elif self.unforced and self.unforced:
            loss1 = nn.functional.mse_loss(y_pred[:, 0:1], e) # signal loss
            loss2 = nn.functional.mse_loss(y_pred[:, 1:2], x-e) # variability loss. ens member - ens mean
            loss3 = nn.functional.mse_loss(y_pred[:, 0:1]+y_pred[:, 1:2], x) # reconstruction loss
            loss = loss1+loss2+loss3
            self.log_dict({'train_loss_signal':loss1, 'step':self.current_epoch})
            self.log_dict({'train_loss_variability':loss2, 'step':self.current_epoch})
            self.log_dict({'train_loss_recon':loss3, 'step':self.current_epoch})
        else:
            loss = nn.functional.mse_loss(y_pred, y)
            # loss = nn.functional.l1_loss(y_pred, y)
        loss_ens = nn.functional.mse_loss(y_pred, e)
        # self.log('train_loss', loss, on_epoch=True, on_step=False)
        # self.log('train_loss', loss)
        self.log_dict({'train_loss':loss, 'step':self.current_epoch})
        self.log_dict({'train_loss_ens':loss_ens, 'step':self.current_epoch})
        return loss
    
    def validation_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        x, y, e = batch
        y_pred = self(x) # batch, 1, time, lat, lon
        if self.signal and not self.unforced:
            loss = nn.functional.mse_loss(y_pred, e)
        elif self.signal and self.unforced:
            loss1 = nn.functional.mse_loss(y_pred[:, 0:1], e) # signal loss
            loss2 = nn.functional.mse_loss(y_pred[:, 1:2], x-e) # variability loss. ens member - ens mean
            loss3 = nn.functional.mse_loss(y_pred[:, 0:1]+y_pred[:, 1:2], x) # reconstruction loss
            loss = loss1+loss2+loss3
            self.log_dict({'val_loss_signal':loss1, 'step':self.current_epoch})
            self.log_dict({'val_loss_variability':loss2, 'step':self.current_epoch})
            self.log_dict({'val_loss_recon':loss3, 'step':self.current_epoch})
        else:
            loss = nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        loss = nn.functional.mse_loss(y_pred, e)
        self.log('ens_loss', loss, on_epoch=True, on_step=False)
        loss = nn.functional.mse_loss(y, e)
        # self.log('ens_loss_target', loss, on_epoch=True, on_step=False)
        self.log_dict({'ens_loss_target':loss, 'step':self.current_epoch})
        
        if batch_idx==0:
            x1 = x.to('cpu').data.numpy()[0, 0, 1]
            yy = y.to('cpu').data.numpy()[0 ,0, 1]
            x2 = y_pred.to('cpu').data.numpy()[0, 0, 1]
            ee = e.to('cpu').data.numpy()[0, 0, 1]
            if self.signal:
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
            else:
                figure = generate_plot(x1=x1, x2=yy, x3=x2, e=ee)
            # x1: input, x2: target, x3: output, e: ensemble mean
            tensorboard_logger.add_figure('val', 
                                          figure, 
                                          global_step=self.current_epoch)
            
    def predict_step(self, batch, batch_idx):
        x, y, e = batch
        print(x.shape, y.shape, e.shape)
        y_pred = self(x) # batch, 1, time, lat, lon

        if self.signal and self.unforced:
            pred_ens = y_pred[:, 0:1] # signal
            pred_var = y_pred[:, 1:2] # variability
            return x, pred_ens, pred_var, e
        else:
            return x, y_pred, e