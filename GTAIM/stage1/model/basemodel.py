import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from model.blocks import ResBlock, TransformerEncoderLayer, PositionalEncoding

from utils.utilities import Console
# from model.pointtransformer.pointtransformer_semseg import pointtransformer_feature_extractor as PointTransformerEnc
import random
from torch.nn.parameter import Parameter
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import math


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
        # print("sdf", sdf.shape)
        fsdf = self.sdf_mlp(sdf)

        return fsdf




class MotionGRU_S1(nn.Module):
    def __init__(self, config):
        super(MotionGRU_S1, self).__init__()
        #his 25
        nx = config.motion_shape
        ny = config.motion_shape
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



    def encode_x(self, x):
        # X S, B 162
        # fs B 128
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x


    def forward(self, x):

        h_x = self.encode_x(x)

        return h_x


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
        self.p = 0.5
        self.dp1 = nn.Dropout3d(p = self.p)
        self.dp2 = nn.Dropout3d(p = self.p)
        self.dp3 = nn.Dropout3d(p = self.p)
        self.dp4 = nn.Dropout3d(p = self.p)
        self.dp5 = nn.Dropout3d(p = self.p)
        self.dp6 = nn.Dropout3d(p = self.p)
        self.dp7 = nn.Dropout3d(p = self.p)


        self.conv0 = ConvBnReLU3D(1, 16, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)

        self.conv1 = ConvBnReLU3D(16, 32,stride=2, kernel_size=3, pad=1)
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)

        self.conv4 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        
        self.conv7 =  ConvBnReLU3D(16, 32, kernel_size=3, pad=1, stride= 2)
        
        self.conv7a = nn.MaxPool3d(3, stride=2)
       
        
        self.conv8 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1, stride= 2)
        
        self.conv8a = nn.MaxPool3d(3, stride=2)

        self.conv9 = ConvBnReLU3D(64, 128, kernel_size=3, pad=1, stride= 2)
        
        self.conv9a = nn.MaxPool3d(3, stride=2)

    def forward(self, x):
        # print("x", x.shape) B, 1, 126, 126, 126
        conv0 = self.conv0a(self.conv0(x))
        conv0 = self.dp1(conv0)
        # print("conv0", conv0.shape)
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        conv2 = self.dp2(conv2)
        # print("conv2", conv2.shape)

        # conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

        conv4 = self.conv4(conv2)
        conv4 = self.dp3(conv4)
        # print("conv4", conv4.shape)
        # print("v5", v5.shape)
        conv5 = conv2+self.conv5(conv4)
        conv5 = self.dp4(conv5)
        # print("conv5", conv5.shape)
        # print("v6", v6.shape)
        conv6 = conv0+self.conv6(conv5)
        conv6 = self.dp5(conv6)
        # conv6 torch.Size([16, 16, 63, 63, 63])
        # print("conv6", conv6.shape)
        conv7 = self.conv7a(self.conv7(conv6))
        conv7 = self.dp6(conv7)
        # print("conv7", conv7.shape)
        conv8 = self.conv8a(self.conv8(conv7))
        conv8 = self.dp7(conv8)
        # print("conv8", conv8.shape)
        conv9 = self.conv9a((self.conv9(conv8)))
        # print("conv9", conv9.squeeze().shape)
        return conv9.squeeze()

# class SceneNet(nn.Module):
#     def __init__(self, config):
#         super(SceneNet, self).__init__()



#         self.conv0 = ConvBnReLU3D(1, 16, kernel_size=3, pad=1)
#         self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)

#         self.conv1 = ConvBnReLU3D(16, 32,stride=2, kernel_size=3, pad=1)
#         self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
#         self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
#         # self.conv3 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
#         self.conv4 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
#         # self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
#         # conv5 torch.Size([B, 32, 63, 63, 63])
#         self.conv5 = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True))

#         self.conv6 = nn.Sequential(
#             nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True))
#         # 125
#         self.conv7 =  ConvBnReLU3D(16, 32, kernel_size=3, pad=1, stride= 2)
#         # 62
#         self.conv7a = nn.MaxPool3d(3, stride=2)
       
#         # 31
#         self.conv8 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1, stride= 2)
#         # 15
#         self.conv8a = nn.MaxPool3d(3, stride=2)
#         # 8
#         # self.conv8b = nn.MaxPool3d(3, stride=2)
#         # 4
#         self.conv9 = ConvBnReLU3D(64, 128, kernel_size=3, pad=1, stride= 2)
#         # 2
#         self.conv9a = nn.MaxPool3d(3, stride=2)
#         # 1
#         # self.conv9b = nn.MaxPool3d(3, stride=2)

#     def forward(self, x):
#         # print("x", x.shape) B, 1, 126, 126, 126
#         conv0 = self.conv0a(self.conv0(x))
 
#         # print("conv0", conv0.shape)
#         conv2 = self.conv2a(self.conv2(self.conv1(conv0)))

#         # print("conv2", conv2.shape)

#         # conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

#         conv4 = self.conv4(conv2)
    
#         # print("conv4", conv4.shape)
#         # print("v5", v5.shape)
#         conv5 = conv2+self.conv5(conv4)
     
#         # print("conv5", conv5.shape)
#         # print("v6", v6.shape)
#         conv6 = conv0+self.conv6(conv5)

#         # conv6 torch.Size([16, 16, 63, 63, 63])
#         # print("conv6", conv6.shape)
#         conv7 = self.conv7a(self.conv7(conv6))

#         # print("conv7", conv7.shape)
#         conv8 = self.conv8a(self.conv8(conv7))
  
#         # print("conv8", conv8.shape)
#         conv9 = self.conv9a((self.conv9(conv8)))
#         # print("conv9", conv9.squeeze().shape)
#         return conv9.squeeze()







class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_H(nn.Module):
    def __init__(self, config, num_stage=12):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_H, self).__init__()
        input_feature = config.dct_n + 128 + 128 
        hidden_feature = config.hidden_feature_gcn
        p_dropout = config.p_dropout
        node_n = config.hsdf_in
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, config.dct_n, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, sdf_in, fs,fh):
        B, N, L = sdf_in.shape
       
        fs = fs.unsqueeze(1).repeat(1,N,1)
        fh = fh.unsqueeze(1).repeat(1,N,1)
    
     
        x = sdf_in
        x_in = torch.cat([sdf_in,fs,fh], dim = -1)

        y = self.gc1(x_in)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x
        return y



class GCN_S(nn.Module):
    def __init__(self, config, num_stage=12):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_S, self).__init__()
        input_feature = config.dct_n + 128 + 128 
        hidden_feature = config.hidden_feature_gcn
        p_dropout = config.p_dropout
        node_n = config.sdf_in
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, config.dct_n, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, sdf_in, fs,fh):
        B, N, L = sdf_in.shape
       
        fs = fs.unsqueeze(1).repeat(1,N,1)
        fh = fh.unsqueeze(1).repeat(1,N,1)
     
        x = sdf_in
        x_in = torch.cat([sdf_in,fs,fh], dim = -1)

        y = self.gc1(x_in)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x
        return y
