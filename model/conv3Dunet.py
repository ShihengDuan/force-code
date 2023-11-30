import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from .utils import generate_plot, trend, R2Loss
from math import ceil
from .ssim import ssim
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import xarray as xa

# ResConnection
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=kernel_size, stride=stride, 
                               padding=kernel_size//2, bias=False, padding_mode='reflect')
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                               kernel_size=kernel_size, stride=stride, 
                               padding=kernel_size//2, bias=False, padding_mode='reflect')
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
        return out

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels_config, kernel_size, dropout=0.5):
        super(UNet, self).__init__()
        self.depth = len(channels_config)
        self.kernel_size = kernel_size
        self.input_channel = in_channels
        self.output_channel = out_channels
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.drop = nn.Dropout(p=dropout)
        # Encoder
        for i in range(self.depth):
            in_channels = in_channels if i == 0 else channels_config[i - 1]
            out_channels = channels_config[i]
            self.encoder.append(self.make_encoder_block(in_channels, out_channels))
        # Decoder
        for i in range(self.depth - 1, -1, -1):
            in_channels = channels_config[i] + (channels_config[i + 1] if i < self.depth - 1 else channels_config[i])
            in_channels = channels_config[i] + (channels_config[i] if i < self.depth - 1 else 0)
            out_channels = channels_config[i-1] if i>0 else channels_config[0]
            print('Decoder: ', in_channels, out_channels)
            self.decoder.append(self.make_decoder_block(in_channels, out_channels))
        # Final layer
        self.final_conv = self.spatial_constant_block(channels_config[0], self.output_channel)

    def make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            BasicBlock3D(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=self.kernel_size),                 # ResBlock. 
            nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, 
                      stride=2, padding=ceil((self.kernel_size-2)/2)),  # half the input size.
        )
        '''
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, 
                      padding=self.kernel_size // 2, padding_mode='reflect'),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            # nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, 
                      stride=2, padding=ceil((self.kernel_size-2)/2)), # half the input size. 
        )
        '''

    def make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=self.kernel_size, 
                      padding=self.kernel_size // 2, padding_mode='reflect'),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels)
            # nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            # nn.ReLU(inplace=True)
        )
    def spatial_constant_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, 
                      padding=self.kernel_size//2, padding_mode='reflect'),
            # nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        )

    def forward(self, x): # Batch, channels, depth, width, height. 
        encoder_outs = []
        # print(x.shape, ' input size')
        # Encoder
        for i in range(self.depth):
            x = self.encoder[i](x)
            encoder_outs.append(x)
        # bottleneck
        # x = self.decoder[i](x)
        # Decoder
        for i in range(0, self.depth-1):
            x = self.drop(x)
            # print(x.shape, ' before decoder')
            x = self.decoder[i](x)
            encoder_out = encoder_outs[-i - 2]
            w_en = encoder_out.shape[-1]
            h_en = encoder_out.shape[-2]
            w_x = x.shape[-1]
            h_x = x.shape[-2]
            if (w_en!=w_x) or (h_x!=h_en):
                # Pad the smaller tensor to match dimensions
                diff_h = encoder_out.size()[3] - x.size()[3]
                diff_w = encoder_out.size()[4] - x.size()[4]
                pad_h = diff_h // 2
                pad_w = diff_w // 2

                # Adjust padding for odd differences
                pad_h_before = pad_h
                pad_w_before = pad_w
                if diff_h % 2 != 0:
                    pad_h_after = pad_h + 1
                else:
                    pad_h_after = pad_h
                if diff_w % 2 != 0:
                    pad_w_after = pad_w + 1
                else:
                    pad_w_after = pad_w
                padded = nn.functional.pad(x, (pad_w_before, pad_w_after, pad_h_before, pad_h_after))
                # Pad the smaller tensor to match the size of the larger tensor
                x = torch.cat((encoder_out, padded), dim=1)  # Concatenate along the channel dimension
            else:
                x = torch.cat((encoder_out, x), dim=1)
        # x = self.drop(x)
        x = self.decoder[-1](x)
        # print(x.shape, ' before final')
        # Final layer
        x = self.final_conv(x)
        return x
    
class UNetLightning3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, channels_config, kernel_size, lr=1e-3, 
                 signal=False, unforced=False, loss='r2', mask=0):
        super(UNetLightning3D, self).__init__()
        self.save_hyperparameters()
        if loss=='r2':
            self.loss_fn = R2Loss()
        elif loss=='l2':
            self.loss_fn = torch.nn.MSELoss()
        elif loss=='l1':
            self.loss_fn = torch.nn.L1Loss()        
        self.model = UNet(in_channels, out_channels, channels_config, kernel_size)
        if signal and unforced:
            self.model = UNet(in_channels, out_channels*2, channels_config, kernel_size)
        self.lr = lr
        self.signal = signal
        self.unforced = unforced
        if mask>0:
            # self.mask = xa.open_dataarray('~/MyWorkSpace/Force-code/Fingerprint/landmask.nc')
            self.mask = xa.open_dataarray('~/MyWorkSpace/Force-code/Fingerprint/REGEN/mask.nc')
            self.mask = torch.from_numpy(self.mask.data).float()
            self.mask = self.mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, H, W
        else:
            self.mask = None
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
        # return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def training_step(self, batch, batch_idx):
        x, y, e = batch
        if self.mask is not None:
            self.extend_mask = self.mask.expand(x.size())
            self.extend_mask = self.extend_mask.to(self.device)
            x = x*self.extend_mask
        y_pred = self(x)
        if self.signal and (not self.unforced): # predict ensemble signal. 
            loss1 = self.loss_fn(y_pred, e)
            self.log('train_loss_signal', loss1, on_epoch=True, on_step=False)
            loss2 = self.loss_fn(x-y_pred, x-e)
            self.log('train_loss_variability', loss2, on_epoch=True, on_step=False)
            loss = loss1+loss2
        elif self.signal and self.unforced: # predict both. 
            loss1 = self.loss_fn(y_pred[:, 0:1], e) # signal loss
            loss2 = self.loss_fn(y_pred[:, 1:2], x-e) # variability loss. ens member - ens mean
            loss3 = self.loss_fn(y_pred[:, 0:1]+y_pred[:, 1:2], x) # reconstruction loss
            trend_pred = trend(y_pred[:, 0:1]) # signal trend
            trend_target = trend(e)
            loss_trend = self.loss_fn(trend_pred, trend_target)
            loss = loss1+loss2+loss3
            self.log('train_loss_signal', loss1, on_epoch=True, on_step=False)
            self.log('train_loss_variability', loss2, on_epoch=True, on_step=False)
            self.log('train_loss_recon', loss3, on_epoch=True, on_step=False)
            self.log('train_loss_trend', loss_trend, on_epoch=True, on_step=False)
        elif not(self.signal) and self.unforced: # predict variability instead of signal
            loss1 = self.loss_fn(y_pred, x-e, mask=self.extend_mask)
            loss2 = self.loss_fn(x-y_pred, e, mask=self.extend_mask)
            trend_pred = trend(x-y_pred)
            trend_target = trend(e)
            loss_trend = self.loss_fn(trend_pred, trend_target)
            # loss = loss1+loss_trend
            loss = loss1+loss2
            self.log('train_loss_variability', loss1, on_epoch=True, on_step=False)
            self.log('train_loss_signal', loss2, on_epoch=True, on_step=False)
        else: # noise to noise. 
            loss = self.loss_fn(y_pred, y)
            # loss = nn.functional.l1_loss(y_pred, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        tensorboard_logger = self.logger.experiment
        if (batch_idx==0) and (self.current_epoch%3==0):
            x1 = x.to('cpu').data.numpy()[0, 0, 1]
            yy = y.to('cpu').data.numpy()[0 ,0, 1]
            x2 = y_pred.to('cpu').data.numpy()[0, 0, 1]
            ee = e.to('cpu').data.numpy()[0, 0, 1]
            if self.signal and (not self.unforced): # predict ensemble signal. 
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('train-signal', 
                                          figure, 
                                          global_step=self.current_epoch)
            elif self.signal and self.unforced: # predict variability and signal. 
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('train-signal', 
                                          figure, 
                                          global_step=self.current_epoch)
                variability_output = y_pred.to('cpu').data.numpy()[0, 1, 1]
                figure = generate_plot(x1=x1, x2=x1-ee, x3=variability_output, e=ee)
                tensorboard_logger.add_figure('train-variability', 
                                          figure, 
                                          global_step=self.current_epoch)
            elif (not self.signal) and self.unforced: # predict variability
                figure = generate_plot(x1=x1, x2=x1-ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('train-variability', 
                                          figure, 
                                          global_step=self.current_epoch)
            # x1: input, x2: target, x3: output, e: ensemble mean
            # SSIM:
            '''
            x1 = x.to('cpu')[:, 0, :] # batch, channel, time. 
            yy = y.to('cpu')[: ,0, :]
            x2 = y_pred.to('cpu')[: ,0, :]
            lat, lon = x2.shape[-1], x2.shape[-2]
            ee = e.to('cpu')[: ,0, :]
            if self.signal and (not self.unforced): # predict ensemble signal. 
                x2 = x2.reshape(-1, 1, lat, lon)
                ee = ee.reshape(-1, 1, lat, lon)
                # Normalize
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                ee = (ee-ee.min())/(ee.max()-ee.min())
                ssim_val = ssim(x2, ee, data_range=1)
                self.log('train_ssim_signal', ssim_val, on_epoch=True, on_step=False)
            elif self.signal and self.unforced:
                var_true = yy-ee
                x2 = x2.reshape(-1, 1, lat, lon)
                ee = ee.reshape(-1, 1, lat, lon)
                # Normalize
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                ee = (ee-ee.min())/(ee.max()-ee.min())
                ssim_val = ssim(x2, ee, data_range=1)
                self.log('train_ssim_signal', ssim_val, on_epoch=True, on_step=False)
                var = y_pred.to('cpu')[:, 1, :]
                var_true = var_true.reshape(-1, 1, lat, lon)
                var = var.reshape(-1, 1, lat, lon)
                var = (var-var.min())/(var.max()-var.min())
                var_true = (var_true-var_true.min())/(var_true.max()-var_true.min())
                ssim_val = ssim(var, var_true, data_range=1)
                self.log('train_ssim_variability', ssim_val, on_epoch=True, on_step=False)
            elif (not self.signal) and self.unforced:
                x2 = x2.reshape(-1, 1, lat, lon)
                var_true = yy-ee
                var_true = var_true.reshape(-1, 1, lat, lon)
                x2 = x2.reshape(-1, 1, lat, lon)
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                var_true = (var_true-var_true.min())/(var_true.max()-var_true.min())
                ssim_val = ssim(x2, var_true, data_range=1)
                self.log('train_ssim_variability', ssim_val, on_epoch=True, on_step=False)
            '''
        return loss
    
    def validation_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        x, y, e = batch
        y_pred = self(x) # batch, 1, time, lat, lon
        if self.signal and not self.unforced:
            loss = self.loss_fn(y_pred, e)
            self.log('val_loss_signal', loss, on_epoch=True, on_step=False)
        elif self.signal and self.unforced:
            loss1 = self.loss_fn(y_pred[:, 0:1], e) # signal loss
            loss2 = self.loss_fn(y_pred[:, 1:2], x-e) # variability loss. ens member - ens mean
            loss3 = self.loss_fn(y_pred[:, 0:1]+y_pred[:, 1:2], x) # reconstruction loss
            loss = loss1+loss2+loss3
            self.log('val_loss_signal', loss1, on_epoch=True, on_step=False)
            self.log('val_loss_variability', loss2, on_epoch=True, on_step=False)
            self.log('val_loss_recon', loss3, on_epoch=True, on_step=False)
            
            trend_pred = trend(y_pred[:, 0:1])
            trend_target = trend(e)
            trend_target = trend_target.data.cpu().numpy()
            trend_pred = trend_pred.data.cpu().numpy()
            r2 = r2_score(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r2-signal', r2, on_epoch=True, on_step=False)
            r, p = pearsonr(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r-signal', r, on_epoch=True, on_step=False)
            trend_pred = trend(x-y_pred[:, 1:2])
            trend_pred = trend_pred.data.cpu().numpy()
            r2 = r2_score(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r2-variability', r2, on_epoch=True, on_step=False)
            r, p = pearsonr(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r-variability', r, on_epoch=True, on_step=False)

        elif (not self.signal) and self.unforced: # y_pred is variability. 
            loss1 = self.loss_fn(y_pred, x-e)
            self.log('val_loss_variability', loss1, on_epoch=True, on_step=False)
            loss2 = self.loss_fn(x-y_pred, e)
            self.log('val_loss_signal', loss2, on_epoch=True, on_step=False)
            trend_pred = trend(x-y_pred)
            trend_target = trend(e)
            loss_trend = self.loss_fn(trend_pred, trend_target)
            self.log('val_loss_trend', loss_trend, on_epoch=True, on_step=False)
            trend_target = trend_target.data.cpu().numpy()
            trend_pred = trend_pred.data.cpu().numpy()
            r2 = r2_score(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r2-variability', r2, on_epoch=True, on_step=False)
            r, p = pearsonr(trend_target.flatten(), trend_pred.flatten())
            self.log('val_trend_r-variability', r, on_epoch=True, on_step=False)
            loss = loss1+loss2
        else:
            loss = self.loss_fn(y_pred, y)
            self.log('val_loss', loss, on_epoch=True, on_step=False)
        
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        
        if batch_idx==20:
            x1 = x.to('cpu').data.numpy()[0, 0, -1] # batch, channel, time. 
            yy = y.to('cpu').data.numpy()[0 ,0, -1]
            x2 = y_pred.to('cpu').data.numpy()[0, 0, -1]
            ee = e.to('cpu').data.numpy()[0, 0, -1]
            if self.signal and (not self.unforced): # predict ensemble signal. 
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('val-signal', 
                                          figure, 
                                          global_step=self.current_epoch)
            elif self.signal and self.unforced: # predict variability and signal. 
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('val-signal', 
                                          figure, 
                                          global_step=self.current_epoch)
                variability_output = y_pred.to('cpu').data.numpy()[0, 1, -1]
                figure = generate_plot(x1=x1, x2=yy-ee, x3=variability_output, e=ee)
                tensorboard_logger.add_figure('val-variability', 
                                          figure, 
                                          global_step=self.current_epoch)
            elif (not self.signal) and self.unforced: # predict variability
                figure = generate_plot(x1=x1, x2=x1-ee, x3=x2, e=ee)
                tensorboard_logger.add_figure('val-variability', 
                                          figure, 
                                          global_step=self.current_epoch)
            # SSIM:
            '''
            x1 = x.to('cpu')[:, 0, :] # batch, channel, time. 
            yy = y.to('cpu')[: ,0, :]
            x2 = y_pred.to('cpu')[: ,0, :]
            lat, lon = x2.shape[-1], x2.shape[-2]
            ee = e.to('cpu')[: ,0, :]
            if self.signal and (not self.unforced): # predict ensemble signal. 
                x2 = x2.reshape(-1, 1, lat, lon)
                ee = ee.reshape(-1, 1, lat, lon)
                # Normalize
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                ee = (ee-ee.min())/(ee.max()-ee.min())
                ssim_val = ssim(x2, ee, data_range=1)
                self.log('val_ssim_signal', ssim_val, on_epoch=True, on_step=False)
            elif self.signal and self.unforced:
                var_true = yy-ee
                x2 = x2.reshape(-1, 1, lat, lon)
                ee = ee.reshape(-1, 1, lat, lon)
                # Normalize
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                ee = (ee-ee.min())/(ee.max()-ee.min())
                ssim_val = ssim(x2, ee, data_range=1)
                self.log('val_ssim_signal', ssim_val, on_epoch=True, on_step=False)
                var = y_pred.to('cpu')[:, 1, :]
                var_true = var_true.reshape(-1, 1, lat, lon)
                var = var.reshape(-1, 1, lat, lon)
                var = (var-var.min())/(var.max()-var.min())
                var_true = (var_true-var_true.min())/(var_true.max()-var_true.min())
                ssim_val = ssim(var, var_true, data_range=1)
                self.log('val_ssim_variability', ssim_val, on_epoch=True, on_step=False)
            elif (not self.signal) and self.unforced:
                x2 = x2.reshape(-1, 1, lat, lon)
                var_true = yy-ee
                var_true = var_true.reshape(-1, 1, lat, lon)
                x2 = x2.reshape(-1, 1, lat, lon)
                x2 = (x2-x2.min())/(x2.max()-x2.min())
                var_true = (var_true-var_true.min())/(var_true.max()-var_true.min())
                ssim_val = ssim(x2, var_true, data_range=1)
                self.log('val_ssim_variability', ssim_val, on_epoch=True, on_step=False)
            '''
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