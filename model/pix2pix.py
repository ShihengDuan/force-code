from torch import nn
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

from .conv3Dunet import UNet
from .utils import generate_plot, trend, R2Loss

class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel, strides, padding)
        if batchnorm:
            self.bn = nn.BatchNorm3d(out_channels)
        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class PatchGAN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 4, batchnorm=False)
        self.d2 = DownSampleConv(4, 8)
        self.d3 = DownSampleConv(8, 32)
        self.final = nn.Conv3d(32, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.final(x)
        x = self.sig(x)
        return x

class WasDis(nn.Module):  # output unbounded values.
    def __init__(self, input_channels, node=32*8*9): # node: flatten size. 
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 4, batchnorm=False)
        self.d2 = DownSampleConv(4, 8)
        self.d3 = DownSampleConv(8, 16)
        self.linear = nn.Linear(node, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class Pix2Pix(pl.LightningModule):
    def __init__(self, in_channels, out_channels, channels_config, kernel_size, 
                lr_gen=5e-4, lr_dis=5e-5, was=False):
        super().__init__()
        self.save_hyperparameters()
        self.gen = UNet(in_channels, out_channels, channels_config, kernel_size, dropout=0.5)
        self.was = was
        if not was:
            self.disc = PatchGAN(in_channels+out_channels)
        else:
            self.disc = WasDis(in_channels+out_channels)
        self.adversarial_loss = nn.BCELoss()
        # self.recon_loss = nn.L1Loss()
        self.recon_loss = R2Loss()
        self.display_step = 4
        self.size = 128
        self.stride = 64

    def _gen_step(self, variability, ens, anomaly):
        pred_var = self.gen(anomaly) # batch, channel, time, lat, lon. 
        var_logits = self.disc(pred_var, anomaly)
        ens_logits = self.disc(anomaly-pred_var, anomaly)
        if self.was:
            # G increase fake score
            adverserial_loss = -torch.mean(var_logits)
        else:
            adverserial_loss_var = self.adversarial_loss(
                var_logits, torch.ones_like(var_logits))
            adverserial_loss_ens = self.adversarial_loss(
                ens_logits, torch.ones_like(ens_logits))
        recon_loss_var = self.recon_loss(pred_var, anomaly)
        recon_loss_ens = self.recon_loss(anomaly-pred_var, ens)
        return adverserial_loss_var, adverserial_loss_ens, recon_loss_var, recon_loss_ens

    def _unet_step(self, variability, ens, anomaly):
        pred_var = self.gen(anomaly)
        recon_loss1 = self.recon_loss(pred_var, variability)
        recon_loss2 = self.recon_loss(anomaly-pred_var, ens)
        loss = recon_loss1 + recon_loss2
        return loss

    def _disc_step(self, variability, ens, anomaly): # real=variability, condition=anomaly
        pred_var = self.gen(anomaly).detach()
        fake_logits = self.disc(pred_var, anomaly)
        real_logits = self.disc(variability, anomaly)
        fake_loss_var = self.adversarial_loss(
            fake_logits, torch.zeros_like(fake_logits))
        real_loss_var = self.adversarial_loss(
            real_logits, torch.ones_like(real_logits))
        # ensemble signal
        pred_ens = anomaly-pred_var
        fake_logits = self.disc(pred_ens, anomaly)
        real_logits = self.disc(ens, anomaly)
        fake_loss_ens = self.adversarial_loss(
            fake_logits, torch.zeros_like(fake_logits))
        real_loss_ens = self.adversarial_loss(
            real_logits, torch.ones_like(real_logits))
        return (real_loss_var+fake_loss_var+real_loss_ens+fake_loss_ens)/4

    def _was_disc_step(self, real, condition):
        fake = self.gen(condition).detach()
        fake_logits = self.disc(fake, condition)
        real_logits = self.disc(real, condition)
        # D decrease fake score
        return -torch.mean(real_logits) + torch.mean(fake_logits)

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_dis = self.hparams.lr_dis
        gen_opt = torch.optim.Adam(lr=lr_gen, params=self.gen.parameters())
        if self.was:
            disc_opt = torch.optim.RMSprop(
                lr=5e-5, params=self.disc.parameters())
        else:
            disc_opt = torch.optim.Adam(
                lr=lr_dis, params=self.disc.parameters())

        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        tensorboard_logger = self.logger.experiment
        # Batch: x, y, e
        x, y, e = batch # x=anomaly, e=ens.  
        var = x-e
        if optimizer_idx == 0:
            if self.was:
                loss = self._was_disc_step(var, x) # real, condition. 
            else:
                loss = self._disc_step(variability=var, ens=e, anomaly=x)
            self.log('D loss ', loss, on_epoch=True, on_step=False)
        elif optimizer_idx == 1:
            adverserial_loss_var, adverserial_loss_ens, recon_loss_var, recon_loss_ens = self._gen_step(variability=var, ens=e, anomaly=x)
            self.log('recon_loss_var', recon_loss_var, on_epoch=True, on_step=False)
            self.log('recon_loss_ens', recon_loss_ens, on_epoch=True, on_step=False)
            self.log('adverserial_loss_var', adverserial_loss_var,
                        on_epoch=True, on_step=False)
            self.log('adverserial_loss_ens', adverserial_loss_ens,
                        on_epoch=True, on_step=False)
            loss = adverserial_loss_ens + adverserial_loss_var + recon_loss_var + recon_loss_ens
            self.log('G loss ', loss, on_epoch=True, on_step=False)
        if self.was:  # weight clipping
            for p in self.disc.parameters():
                p.data.clamp_(-0.01, 0.01)
        if self.current_epoch % self.display_step == 0 and batch_idx == 0 and optimizer_idx == 1:
            fake = self.gen(x).detach()
            x1 = x.to('cpu').data.numpy()[0, 0, 1]
            x2 = fake.to('cpu').data.numpy()[0, 0, 1]
            ee = e.to('cpu').data.numpy()[0, 0, 1]
            figure = generate_plot(x1=x1, x2=x1-ee, x3=x2, e=ee) # x1: input, x2: target, x3: output, e: ensemble mean
            tensorboard_logger.add_figure('train-variability', 
                                        figure, 
                                        global_step=self.current_epoch)
            # x1: input, x2: target, x3: output, e: ensemble mean
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, e = batch
        out = self.gen(x)
        var = x-e
        return x, out, var

    def validation_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        # Batch: x, y, e
        x, y, e = batch # x=anomaly, e=ens.  
        var = x-e
        if self.was:
            loss = self._was_disc_step(var, x) # real, condition. 
        else:
            loss = self._disc_step(variability=var, ens=e, anomaly=x)
        self.log('D loss val', loss, on_epoch=True, on_step=False)

        adverserial_loss_var, adverserial_loss_ens, recon_loss_var, recon_loss_ens = self._gen_step(variability=var, ens=e, anomaly=x)
        self.log('recon_loss_var val', recon_loss_var, on_epoch=True, on_step=False)
        self.log('recon_loss_ens val', recon_loss_ens, on_epoch=True, on_step=False)
        self.log('adverserial_loss_var val', adverserial_loss_var,
                    on_epoch=True, on_step=False)
        self.log('adverserial_loss_ens val', adverserial_loss_ens,
                    on_epoch=True, on_step=False)
        loss = adverserial_loss_ens + adverserial_loss_var + recon_loss_var + recon_loss_ens
        self.log('G loss val', loss, on_epoch=True, on_step=False)
        if self.current_epoch % self.display_step == 0 and batch_idx == 0:
            fake = self.gen(x).detach()
            x1 = x.to('cpu').data.numpy()[0, 0, 1]
            x2 = fake.to('cpu').data.numpy()[0, 0, 1]
            ee = e.to('cpu').data.numpy()[0, 0, 1]
            figure = generate_plot(x1=x1, x2=x1-ee, x3=x2, e=ee) # x1: input, x2: target, x3: output, e: ensemble mean
            tensorboard_logger.add_figure('val-variability', 
                                        figure, 
                                        global_step=self.current_epoch)