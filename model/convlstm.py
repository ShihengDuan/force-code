import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from .utils import generate_plot

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        # print(combined.shape, ' combined')
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_channels = [input_channels] + hidden_channels[:-1]  # For input channel matching
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cells = []
        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            # print(self.input_channels[i], self.hidden_channels[i], ' ConvIn&Out')
            cells.append(cell)
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        '''if self.batch_first:
            x = x.permute(0, 4, 1, 2, 3)  # Change to (batch_size, sequence_length, channels, height, width)'''

        batch_size, sequence_length, _, height, width = x.size() # B, S, C, H, W
        # print(batch_size, sequence_length, height, width)
        h, c = [torch.zeros(batch_size, hidden, height, width).to(x.device) for hidden in self.hidden_channels], \
               [torch.zeros(batch_size, hidden, height, width).to(x.device) for hidden in self.hidden_channels]

        outputs = []

        for t in range(sequence_length):
            input_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.cells):
                # print(input_t.shape, h[i].shape, ' Input And HiddenState')
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]

            outputs.append(h[-1])  # Use the final hidden state as output

        outputs = torch.stack(outputs, dim=1)

        return outputs

class BidirectionalConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, kernel_size, num_layers, batch_first=True):
        super(BidirectionalConvLSTM, self).__init__()
        self.down_cov = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.forward_lstm = ConvLSTM(hidden_channels[0], hidden_channels, kernel_size, num_layers)
        self.reverse_lstm = ConvLSTM(hidden_channels[0], hidden_channels, kernel_size, num_layers)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[-1]*2, hidden_channels[-1]*2 // 2, kernel_size=2, stride=2),
            nn.Conv2d(hidden_channels[-1]*2 // 2, hidden_channels[-1]*2 // 2, kernel_size=kernel_size, 
                      padding=kernel_size // 2),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1]*2 // 2, out_channels, kernel_size, padding=kernel_size //2),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size //2),
                        )
        self.out_channels = out_channels
    def forward(self, x):
        # print(x.shape, ' input shape') # B, T, C, Lat, Lon
        b, t, c, lat, lon = x.size()
        x = x.reshape(b*t, c, lat, lon)
        x = self.down_cov(x)
        _, c, lat, lon = x.size()
        # print(x.shape, ' After downcov')
        x = x.reshape(b, t, -1, lat, lon)
        # print(x.shape, ' input to lstm')
        forward_output = self.forward_lstm(x)
        reverse_output = self.reverse_lstm(torch.flip(x, [1]))  # Reverse the sequence along the time dimension
        reverse_output = torch.flip(reverse_output, [1])  # Flip the reversed output back
        # print(forward_output.shape, reverse_output.shape)
        x = torch.cat((forward_output, reverse_output), dim=2) # B, S, C, H, W. 
        b, s, c, w, h = x.size()
        x = x.reshape(s*b, c, w, h)
        x = self.up_conv(x)
        x = self.out_conv(x)
        _, c, w, h = x.size()
        x = x.reshape(b, s, c, w, h)
        x = x.permute(0, 2, 1, 3, 4)
        return x

class BidConvLSTM3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_layers, lr, signal=False):
        super(BidConvLSTM3D, self).__init__()
        self.model = BidirectionalConvLSTM(in_channels, hidden_channels, out_channels, kernel_size, num_layers)
        self.lr = lr
        self.signal=signal
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
        # return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def training_step(self, batch, batch_idx):
        x, y, e = batch # B, C, time, lat, lon
        x = x.permute(0, 2, 1, 3, 4)
        y_pred = self(x) # B, C, S, W, H
        if self.signal:
            loss = nn.functional.mse_loss(y_pred, e)
        else:
            loss = nn.functional.mse_loss(y_pred, y)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        x, y, e = batch
        x = x.permute(0, 2, 1, 3, 4)
        y_pred = self(x) # batch, 1, time, lat, lon
        if self.signal:
            loss = nn.functional.mse_loss(y_pred, e)
        else:
            loss = nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        loss = nn.functional.mse_loss(y_pred, e)
        self.log('ens_loss', loss, on_epoch=True, on_step=False)
        loss = nn.functional.mse_loss(y, e)
        self.log('ens_loss_target', loss, on_epoch=True, on_step=False)
        if batch_idx==0:
            x1 = x.to('cpu').data.numpy()[0, 0, 0]
            yy = y.to('cpu').data.numpy()[0 ,0, 0]
            x2 = y_pred.to('cpu').data.numpy()[0, 0, 0]
            
            ee = e.to('cpu').data.numpy()[0, 0, 0]
            if self.signal:
                figure = generate_plot(x1=x1, x2=ee, x3=x2, e=ee)
            else:
                figure = generate_plot(x1=x1, x2=yy, x3=x2, e=ee)
            # x1: input, x2: target, x3: output, e: ensemble mean
            tensorboard_logger.add_figure('val', 
                                          figure, 
                                          global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1, 3, 4)
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('test_loss', loss)
    

