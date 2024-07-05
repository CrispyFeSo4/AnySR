# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace
import argparse

import torch
import torch.nn as nn
import random

from models import register
import time
import math
import torch.nn.functional as F
from argparse import Namespace


torch.autograd.set_detect_anomaly(True)
import models.arch_util as arch_util
from models.arch_util import USConv2d
from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, us=[False, False]):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mult = 1.0
        self.us = us

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        out_channels = self.out_channels
        if self.us[1]: 
            if self.width_mult is not None:
                out_channels = int(self.out_channels * self.width_mult)
            else:
                self.width_mult = 1.0  
                out_channels = self.out_channels

        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias

        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class USConv2d_fc2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, us=[False, False]):
        super(USConv2d_fc2, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mult = 1.0
        self.us = us

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        out_channels = self.out_channels
        if self.us[1]: 
            if self.width_mult is not None:
                out_channels = int(self.out_channels//2 * self.width_mult) * 2
            else:
                self.width_mult = 1.0  
                out_channels = self.out_channels

        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias

        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class RDB_USConv2d_former_first(nn.Module): 
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_USConv2d_former_first, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            USConv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1,bias=True, us=[False,True]),
            nn.ReLU()
        ]) 
        
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1) 

class RDB_USConv2d_former(nn.Module): 
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_USConv2d_former, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            USConv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1,bias=True, us=[True,True]),
            nn.ReLU()
        ])        

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1) 
    
class RDB_USConv2d_latter(nn.Module): 
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_USConv2d_latter, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            USConv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1,bias=True, us=[True,False]),
            nn.ReLU()
        ])         

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1) 

class RDB_Conv(nn.Module): 
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ]) 

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1) 


class RDB_any(nn.Module): 
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB_any, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers 
        self.C = nConvLayers
        self.G0 = G0
        self.G = G

        convs = []
        for i in range(C//2): # 0,1,2,3
            convs.append(RDB_Conv(G0+i*G,G))

        convs.append(RDB_USConv2d_former_first(G0+(C//2)*G, G)) # 4
        for c in range(C-1-C//2): # c=0, block=5
            if (c)%2 == 1:  # former
                convs.append(RDB_USConv2d_former(G0 + (c+5)*G, G)) 
            else:
                convs.append(RDB_USConv2d_latter(G0 + (c+5)*G, G)) 

        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = USConv2d(G0 + C * G, G0, 1, padding=0, stride=1,bias=True, us=[True,False])
        self.fc1 = USConv2d(2*(G+8), 256, 1, 1, 0, bias=True, us=[True,False])
        self.fc2 = USConv2d_fc2(256, 2*G, 1, 1, 0, bias=True, us=[False,True])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2    

    def set_width(self, width):
        for c in range(self.C): 
            self.convs[c].conv[0].width_mult = width
        self.LFF.width_mult = width
        self.fc1.width_mult = width
        self.fc2.width_mult = width

    def scale_aware_attn(self,feature,r): 
        B,C,H,W = feature.size()
        global_avg_pooled = self.gap(feature)
        feature = global_avg_pooled.view(B, -1) 
        r0 = r.to('cuda')

        max_scale = max(r[0],r[1])
        if max_scale*10%2 == 1:
            flag = -1
        else:
            flag = 1
        r0 = r0 * flag

        r0 = r0.expand(B,-1)

        interleaved_feature = []
        for i in range(0,C,16):# 0-C,step=16
            if i+16 <= C: 
                feature_trunk = feature[:,i:i+16]
                interleaved_feature.append(feature_trunk)
                interleaved_feature.append(r0)

            else:
                remain_channel = C - i
                feature_trunk = feature[:,-remain_channel:]
                interleaved_feature.append(feature_trunk)

        scale_fea = torch.cat(interleaved_feature, dim=1)
        
        res = self.fc1(scale_fea.view(B,-1,1,1)) 
        res = F.relu(res)
        weight = self.fc2(res) 
        
        result = torch.sigmoid(weight)
        return result

    def forward(self, x):
        B = x.shape[0]
        res = self.convs(x) 
        r_info = torch.tensor([self.scale,self.scale2])
        
        former_fea = res[:,:self.G0+6*self.G,:,:]
        attn_fea = res[:,self.G0+6*self.G:,:,:]

        attn_weight = self.scale_aware_attn(attn_fea,r_info).view(B,-1,1,1) 

        out = attn_fea * attn_weight.expand_as(attn_fea)
        res = torch.cat((former_fea,out),dim=1).expand_as(res)

        tmp = self.LFF(res)
        res  = tmp + x
        return res


class RDN_anysr(nn.Module):
    def __init__(self, args):
        super(RDN_anysr, self).__init__()
        self.scale = args.scale[0]
        self.scale2 = self.scale
        G0 = args.G0
        kSize = args.RDNkSize
        self.scale_list = [round(x * 0.1 + 1, 1) for x in range(1, 31)]
        self.args = args
        self.scale_idx = 0

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
       
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.RDBs = nn.ModuleList()
        for i in range(self.D): 
            self.RDBs.append(
                RDB_any(growRate0=G0, growRate=G, nConvLayers=C)
            )
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        max_scale = max(self.scale,self.scale2)

        parser = argparse.ArgumentParser()
        parser.add_argument('--test_only', type=int)
        parser.add_argument('--entire_net', type=int)
        args, _ = parser.parse_known_args()

        if max_scale in self.scale_list[:7]:
            width = 0.7
            if args.test_only == 0: # train
                if random.random() < 0.6:
                    width = 1.0
        elif max_scale in self.scale_list[7:15]:
            width = 0.8
        elif max_scale in self.scale_list[15:22]:
            width = 0.9
        else:
            width = 1.0
        if args.entire_net == 1:
            width = 1

        x = self.sub_mean(x) 
        f__1 = self.SFENet1(x) 
        x = self.SFENet2(f__1)

        RDBs_out = [] 
        for i in range(self.D):
            self.RDBs[i].set_width(width)
            self.RDBs[i].set_scale(self.scale,self.scale2)
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1)) 
        x += f__1 

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


@register('rdn-anysr')
def make_rdn(G0=64, RDNkSize=3, RDNconfig='B',
             scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range

    args.n_colors = 3
    return RDN_anysr(args)
