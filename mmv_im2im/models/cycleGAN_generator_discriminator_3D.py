
## Adapted for neoamos 3D implementation of pix2pix and CycleGAN and davidiommi CycleGAN

##https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging/tree/main/models
# 
# https://github.com/neoamos/3d-pix2pix-CycleGAN/blob/master/models/networks3d.py 

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import monai
import numpy as np


## Define the ResNet Block

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        conv_block=[]
        ## reflection based padding
        conv_block+=[nn.ReplicationPad3d(1)]
        conv_block+=[nn.Conv3d(dim,dim,kernel_size=3,padding=0,bias=use_bias),
                     norm_layer(dim),
                     nn.ReLU(True)]
        if use_dropout:
            conv_block+=[nn.Dropout(0.5)]
        conv_block+=[nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x+self.conv_block(x)
        return out 


## Define the ResNet Generator

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer =nn.InstanceNorm3d, use_dropout = False, n_blocks=6):
        assert(n_blocks >=0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        use_bias=True

        model =[nn.ReplicationPad3d(3),
                nn.Conv3d(input_nc, ngf, kernel_size=7,padding=0,bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]

        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2**i
            model +=[nn.Conv3d(ngf*mult, ngf *mult*2, kernel_size = 3,stride =2, padding = 1, bias=use_bias),
                     norm_layer(ngf *mult*2),
                     nn.ReLU(True)]
        
        mult = 2**n_downsampling

        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult/2),kernel_size=3, stride =2, padding = 1, output_padding = 1, bias= use_bias),
                      norm_layer(int(ngf * mult/2)),
                      nn.ReLU(True)]
        model+= [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


## Define the  PatchGAN Discriminator

class Discriminator(nn.Module):
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer = nn.InstanceNorm3d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        use_bias=True
        
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size = kw, stride = 2, padding = padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=1, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# ## Adapted for neoamos 3D implementation of pix2pix and CycleGAN
# # 
# # https://github.com/neoamos/3d-pix2pix-CycleGAN/blob/master/models/networks3d.py 


