

# Adapted from https://github.com/enochkan/vox2vox/blob/master/models.py

# https://towardsdatascience.com/volumetric-medical-image-segmentation-with-vox2vox-f5350ed2094f


# https://arxiv.org/pdf/2003.13653.pdf Vox2Vox: 3D-GAN for Brain Tumour
#Segmentation

import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetDownward(nn.Module):
    def __init__(self, in_size, out_size, normalize=False, dropout=0.0):
        super(UNetDownward, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid,self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x=torch.cat((x, skip_input), 1)
        x=self.model(x)
        x=nn.functional.pad(x,(1,0,1,0,1,0))
        return x

class UNetUpward(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpward, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetGenerator, self).__init__()
        print("in_channels",in_channels)
        self.down1 = UNetDownward(in_channels, 64, normalize=True)
        self.down2 = UNetDownward(64,128)
        self.down3 = UNetDownward(128,256)
        self.down4 = UNetDownward(256,512)
        self.mid1 = UNetMid(1024, 512, dropout = 0.2)
        self.mid2 = UNetMid(1024, 512, dropout = 0.2)
        self.mid3 = UNetMid(1024, 512, dropout = 0.2)
        self.mid4 = UNetMid(1024, 256, dropout = 0.2)
        self.up1 = UNetUpward(256, 256)
        self.up2 = UNetUpward(512, 128)
        self.up3 = UNetUpward(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m1 = self.mid1(d4, d4)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=1, padding=1,bias=False)]
            if normalization:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=True),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.model(img_input)
        pad = nn.functional.pad(intermediate, pad=(1,0,1,0,1,0))
        return self.final(pad)






# def test():

#     input_nc=1
#     image_size=128
#     Minimum depth to be maintained is 6 and above for the above stride, padding and filter size of (4, 1, 1) respectively
#     depth=64
#     input1=torch.randn(1,input_nc,depth,image_size,image_size)
#     generator_model=UNetGenerator(in_channels=1, out_channels=1)
#     fake_input1 = generator_model(input1)
#     input2=torch.randn(1,input_nc,depth,image_size,image_size)
#     discriminator_model=Discriminator(in_channels=1)
#     preds=discriminator_model(fake_input1,input2)
#     print(preds.shape)

# if __name__ == "__main__":
#     test()









