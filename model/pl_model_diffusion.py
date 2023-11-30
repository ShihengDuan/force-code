import pytorch_lightning as pl
import logging
import os
import typing
from collections import OrderedDict, Sequence

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR
from .networks_diffusion import define_network

class DDPM(pl.LightningModule):
    def __init__(self, in_channel, out_channel, norm_groups, inner_channel,
                 channel_multiplier, attn_res, res_blocks, dropout,
                 diffusion_loss, conditional, init_method,
                 train_schedule, train_n_timestep, train_linear_start, train_linear_end,
                 val_schedule, val_n_timestep, val_linear_start, val_linear_end,
                 finetune_norm, optimizer, amsgrad, learning_rate, checkpoint, resume_state,
                 phase, height, ema_mu=None):
        """
        norm_groups: The number of groups for group normalization.
        inner_channel: Timestep embedding dimension.
        channel_multiplier: A tuple specifying the scaling factors of channels.
        attn_res: A tuple of spatial dimensions indicating in which resolutions to use self-attention layer.
        res_blocks: The number of residual blocks.
        dropout: Dropout probability.
        diffusion_loss: Either l1 or l2.
        finetune_norm: Whetehr to fine-tune or train from scratch.
        """
        super().__init__()
        # print(diffusion_loss)
        self.SR_net = define_network(in_channel, out_channel, norm_groups, inner_channel,
                                         channel_multiplier, attn_res, res_blocks, dropout,
                                         diffusion_loss, conditional, # gpu_ids, distributed,
                                         init_method, height) # returns the diffusion model. 
        self.loss_type = diffusion_loss
        self.learning_rate = learning_rate
        # self.data, self.SR = None, None
        self.checkpoint = checkpoint
        self.resume_state = resume_state
        self.finetune_norm = finetune_norm
        self.phase = phase
        self.optimizer = optimizer
        self.SR_net.set_new_noise_schedule(schedule=train_schedule, n_timestep=train_n_timestep,
                                     linear_start=train_linear_start, linear_end=train_linear_end,
                                     device=self.device)
        self.SR_net.set_loss(device=self.device)
        self.ema = None
        '''if ema_mu is not None:
            self.ema = EMA(mu=ema_mu)'''
        
        # self.set_loss()
        # self.months = []  # A list of months of curent data given by feed_data.

        '''if self.phase == "train":
            self.set_new_noise_schedule(schedule=train_schedule, n_timestep=train_n_timestep,
                                        linear_start=train_linear_start, linear_end=train_linear_end)
        else:
            self.set_new_noise_schedule(schedule=val_schedule, n_timestep=val_n_timestep,
                                        linear_start=val_linear_start, linear_end=val_linear_end)'''
    def configure_optimizers(self):
        optm = self.optimizer(self.SR_net.parameters(), 
                              lr=self.learning_rate, amsgrad=False) # not finetune. 
        # Learning rate schedulers.
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=40000, eta_min=1e-6)
        scheduler = MultiStepLR(optm, milestones=[40000], gamma=0.5)
        return [optm], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        x = batch # x, y, e
        # print(x.keys())
        # self.ema = EMA(mu=self.ema_mu)
        # self.ema.register(self.SR_net)

        loss = self.SR_net(x)
        return {'loss': loss}
    
    def training_step_end(self, train_step_outputs):
        if self.ema is not None:
            self.ema.update(self.SR_net)
        
    def validation_step(self, batch, batch_idx):
        if self.ema is not None:
            original_SR_net = self.ema.ema_copy(self.SR_net)
        x = batch
        loss = self.SR_net(x)
    
        if self.ema is not None:
            self.SR_net = original_SR_net
        
        return {'loss': loss, 'batch': batch}
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([b['loss'] for b in validation_step_outputs], dim=0).mean().item()
        self.log('val_loss', val_loss)
        
        # sr_loss = self.SR_net.loss_func(x_hr, x_hr_hat)
        # sr_loss = sr_loss.mean(dim=0).item()
        
        # if not isinstance(self.trainer.logger, Sequence):
        #     loggers = [self.trainer.logger]
        
        # print(self.trainer.logger)
        '''for logger in self.trainer.logger:
            if hasattr(logger, 'log_csv'):
                logger.log_csv({'val_loss': val_loss}, step=self.current_epoch)
            elif hasattr(logger, 'log_img'):
                if logger.interval != 0 and self.current_epoch > 0 and self.current_epoch % logger.interval == 0:
                    batch = validation_step_outputs[-1]['batch']
                    x = batch['INTERPOLATED'][:8]
                    y_hr = batch['HR'][:8]
                    y_hr_hat = self.SR_net.super_resolution(x)
                    logger.log_img({'x': x.cpu(), 'y': y_hr.cpu(), 'y_hat': y_hr_hat.cpu()}, step=self.current_epoch)
        '''        
    def training_epoch_end(self, train_step_outputs):
        train_loss = torch.stack([b['loss'] for b in train_step_outputs], dim=0).mean().item()
        self.log('val_loss', train_loss)
        
        # if not isinstance(self.trainer.logger, Sequence):
        #     loggers = [self.trainer.logger]
        
        # print(self.trainer.logger)
        for logger in self.trainer.logger:
            if hasattr(logger, 'log_csv'):
                logger.log_csv({'train_loss': train_loss}, step=self.current_epoch)
        
        for logger in self.trainer.logger:
            if hasattr(logger, 'write'):
                logger.write()
                
        # TODO: add other metrics for evaluation
        
    
    