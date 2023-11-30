import torch
import torch.nn as nn
import pytorch_lightning as pl
from matplotlib import pyplot as plt
def generate_plot(x1, x2, e):
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(x1, origin='lower')
        ax = fig.add_subplot(132)
        ax.imshow(x2, origin='lower')
        ax = fig.add_subplot(133)
        ax.imshow(e, origin='lower')
        ax.set_title('ENS')
        return fig

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels_config, kernel_size):
        super(UNet, self).__init__()
        self.depth = len(channels_config)
        self.kernel_size = kernel_size
        self.input_channel = in_channels
        self.output_channel = out_channels
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

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
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(inplace=True)
        )
    def spatial_constant_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        )

    def forward(self, x):
        encoder_outs = []

        # Encoder
        for i in range(self.depth):
            # print(x.shape, ' encoder')
            x = self.encoder[i](x)
            # print(x.shape, ' encoder')
            encoder_outs.append(x)
        # bottleneck
        # x = self.decoder[i](x)
        # Decoder
        for i in range(0, self.depth-1):
            # print(x.shape, ' before decoder')
            x = self.decoder[i](x)
            # print(x.shape, ' after decoder')
            encoder_out = encoder_outs[-i - 2]

            # Pad the smaller tensor to match dimensions
            diff_h = encoder_out.size()[2] - x.size()[2]
            diff_w = encoder_out.size()[3] - x.size()[3]
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
        x = self.decoder[-1](x)
        # print(x.shape, ' before final')
        # Final layer
        x = self.final_conv(x)
        return x
    
class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels, channels_config, kernel_size):
        super(UNetLightning, self).__init__()
        self.model = UNet(in_channels, out_channels, channels_config, kernel_size)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y, e = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        x, y, e = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        if batch_idx==0:
            x1 = x.to('cpu').data.numpy()[0, 0]
            x2 = y_pred.to('cpu').data.numpy()[0, 0]
            ee = e[0, 0].to('cpu')
            figure = generate_plot(x1, x2, ee)
            tensorboard_logger.add_figure('val', 
                                          figure, 
                                          global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('test_loss', loss)
    
