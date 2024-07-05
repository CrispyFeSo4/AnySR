# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

import math
import random
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
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

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ResidualBlock_noBN_anysr(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_anysr, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[False,True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True,False])
        self.fc1 = USConv2d(nf+8, nf*2, 1, 1, 0, bias=True, us=[True,False])
        self.fc2 = USConv2d(nf*2, nf, 1, 1, 0, bias=True, us=[False,True])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.nf = nf

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.fc1, self.fc2], 0.1)
    
    def set_width(self, width):
        self.conv1.width_mult = width
        self.conv2.width_mult = width
        self.fc1.width_mult = width
        self.fc2.width_mult = width
    
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

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
        
        r0 = r0.expand(B, -1) 
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
        identity = x
        b = identity.shape[0]
        out = self.conv1(x) 
        
        r_info = torch.tensor([self.scale,self.scale2])

        attn_weight = self.scale_aware_attn(out,r_info).view(b,-1,1,1) 
        
        out = out * attn_weight.expand_as(out)
        tmp = F.relu(out)
        out = self.conv2(tmp)
        return identity + out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR_anysr(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR_anysr, self).__init__()

        n_resblock = args.n_resblocks
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        self.scale2 = self.scale

        act = nn.ReLU(True)
        self.scale_list = [round(x * 0.1 + 1, 1) for x in range(1, 31)]

        self.args = args
        self.scale_idx = 0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, self.scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range,rgb_mean,rgb_std)
        self.add_mean = MeanShift(args.rgb_range,rgb_mean,rgb_std)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_body = []
        for _ in range(n_resblock):
            m_body.append(ResidualBlock_noBN_anysr(n_feats))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors


    def forward(self, x):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_only', type=int)
        parser.add_argument('--entire_net', type=int)
        args, _ = parser.parse_known_args()

        x = self.sub_mean(x)
        max_scale = max(self.scale,self.scale2)
        if max_scale in self.scale_list[:7]:
            width = 0.5
            if args.test_only == 0: # train
                if random.random() < 0.6:
                    width = 1.0
        elif max_scale in self.scale_list[7:15]:
            width = 0.7
        elif max_scale in self.scale_list[15:22]:
            width = 0.9
        else:
            width = 1.0
        if args.entire_net == 1:
            width = 1

        x = self.head(x)
        res = x

        for i in range(self.args.n_resblocks):
            self.body[i].set_width(width) 
            self.body[i].set_scale(self.scale,self.scale2) 
            res = self.body[i](res)
            
        res = self.body[-1](res)
        res += x
        return res


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-anysr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR_anysr(args)


@register('edsr-anysr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR_anysr(args)
