import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from model.blocks import ResBlock, TransformerEncoderLayer, PositionalEncoding
from pytorch_pretrained_bert.modeling import BertModel
from utils.utilities import Console

import random

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]

class RNN(nn.Module):
    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = torch.zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = torch.zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = torch.zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = torch.zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out

class SDFNet(nn.Module):
    def __init__(self,config):
        super(SDFNet, self).__init__()
        self.sdf_mlp = MLP(config.sdf_in, [128,config.sdf_out], activation='relu')
    
    def forward(self, sdf):

        fsdf = self.sdf_mlp(sdf)

        return fsdf


class MotionGRU(nn.Module):
    def __init__(self, config):
        super(MotionGRU, self).__init__()
        #his 25
        nx = config.motion_shape
        ny = config.motion_shape
        nh = config.nscene_point
        self.ns = ns = 128
        self.nsdf = nsdf = config.sdf_out
        horizon = config.future_len
        self.nx = nx
        self.ny = ny
        # #128
        # self.nz = nz
        #100
        self.horizon = horizon
        #GRU
        self.rnn_type = rnn_type = 'gru'
        #False
        self.x_birnn = x_birnn = False
        #False
        self.e_birnn = e_birnn = False
        #True
        self.use_drnn_mlp = True
        #128
        self.nh_rnn = nh_rnn = 128
        #[300, 200]
        self.nh_mlp = nh_mlp = [300,200]
        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.sdf_rnn = RNN(128, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        # self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        # self.e_mlp = MLP(2*nh_rnn, nh_mlp)
        # self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        # self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nh_rnn + ns + nsdf + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')

        self.h_mlp = MLP(nh, [nh_rnn])
        self.h_rnn = RNN(nh_rnn, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)

    def encode_x(self, x):
        # X S, B 162
        # fs B 128
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x
    

    def encode_sdf(self, sdf):
        sdf = self.sdf_rnn(sdf)
        return sdf
    

    def encode_hsdf(self, hsdf):
        hsdf = self.h_mlp(hsdf)
        hsdf = self.h_rnn(hsdf)
        return hsdf


    def forward(self, x, fs, fsdf, fhsdf):
        # fsdf S, B, sdf_out
        # print("fsdf shape", fsdf.shape)
        h_x = self.encode_x(x)
        # print("hx", h_x.shape) 64 128
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=x.shape[1], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=x.shape[1])

        
        y = []
        # print("x")
        # print(x.shape)
        # print(x[-1,:,:3])
        fsdf = self.encode_sdf(fsdf)
        fhsdf = self.encode_hsdf(fhsdf)
 
        fsdf = fsdf[30:,:,:]
        fhsdf = fhsdf[30:,:,:]
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i

            fsdfi = fsdf[i,:,: ].squeeze()
            fhsdfi = fhsdf[i,:,:].squeeze()
            # print(fsdfi.shape)
            rnn_in = torch.cat([h_x, y_p, fs, fsdfi, fhsdfi], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            output = self.d_out(h)
            
            y_i = output + y_p
            # print(output.shape,y_i.shape)
            # print(i, output[0,:3],y_p[0,:3])
            y.append(y_i)
        y = torch.stack(y)
        return y

    # def forward(self, x, y):
    #     mu, logvar = self.encode(x, y)
    #     z = self.reparameterize(mu, logvar) if self.training else mu
    #     return self.decode(x, z), mu, logvar

    # def sample_prior(self, x):
    #     z = torch.randn((x.shape[1], self.nz), device=x.device)
    #     return self.decode(x, z)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class SceneNet(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):

        return x


class SceneNet(nn.Module):
    def __init__(self, config):
        super(SceneNet, self).__init__()

        self.conv0 = ConvBnReLU3D(1, 16, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)

        self.conv1 = ConvBnReLU3D(16, 32,stride=2, kernel_size=3, pad=1)
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        # self.conv3 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv4 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        # self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        # conv5 torch.Size([B, 32, 63, 63, 63])
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        # 125
        self.conv7 =  ConvBnReLU3D(16, 32, kernel_size=3, pad=1, stride= 2)
        # 62
        self.conv7a = nn.MaxPool3d(3, stride=2)
       
        # 31
        self.conv8 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1, stride= 2)
        # 15
        self.conv8a = nn.MaxPool3d(3, stride=2)
        # 8
        # self.conv8b = nn.MaxPool3d(3, stride=2)
        # 4
        self.conv9 = ConvBnReLU3D(64, 128, kernel_size=3, pad=1, stride= 2)
        # 2
        self.conv9a = nn.MaxPool3d(3, stride=2)
        # 1
        # self.conv9b = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        # print("x", x.shape) B, 1, 126, 126, 126
        conv0 = self.conv0a(self.conv0(x))
        # print("conv0", conv0.shape)
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        # print("conv2", conv2.shape)

        # conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

        conv4 = self.conv4(conv2)
        # print("conv4", conv4.shape)
        # print("v5", v5.shape)
        conv5 = conv2+self.conv5(conv4)
        # print("conv5", conv5.shape)
        # print("v6", v6.shape)
        conv6 = conv0+self.conv6(conv5)
        # conv6 torch.Size([16, 16, 63, 63, 63])
        # print("conv6", conv6.shape)
        conv7 = self.conv7a(self.conv7(conv6))
        # print("conv7", conv7.shape)
        conv8 = self.conv8a(self.conv8(conv7))

        # print("conv8", conv8.shape)
        conv9 = self.conv9a((self.conv9(conv8)))
        # print("conv9", conv9.squeeze().shape)
        return conv9.squeeze()