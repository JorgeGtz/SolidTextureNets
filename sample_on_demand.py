# sample_on_demand.py
#
# Generates a new sample using a trained generator
# The sample is saved as a numpy array with the channels as BGR
# The noise inputs come from a spatially seeded PRNG
#
# Code for the article:
# On Demand Solid Texture Synthesis Using Deep 3D Networks
# J. Gutierrez, J. Rabin, B. Galerne, T. Hurtut
#
# Author: Jorge Gutierrez
# Last modified: 12 Dec 2020
#

import math
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms

import cupy as cp

#generator's convolutional blocks 3D
class Conv_block3D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block3D, self).__init__()

        self.conv1 = nn.Conv3d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv3d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm3d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv3d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm3d(n_ch_out, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

# up-sampling + batch normalization block
class Up_Bn3D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn3D, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm3d(n_ch)

    def forward(self, x):
        # x = self.bn(self.up(x))
        x = self.bn(F.interpolate(x,scale_factor=2, mode='nearest'))
        return x

class Shift3D(nn.Module):
    def __init__(self):
        super(Shift3D, self).__init__()

    def forward(self, x, shift):
        x = torch.cat((x[:,:,shift[0]:,:,:],x[:,:,:shift[0],:,:]),2)
        x = torch.cat((x[:,:,:,shift[1]:,:],x[:,:,:,:shift[1],:]),3)
        x = torch.cat((x[:,:,:,:,shift[2]:],x[:,:,:,:,:shift[2]]),4)
        return x

class MultiScaleGen3D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8, n_h=0, n_w=0, n_d=0):
        super(MultiScaleGen3D, self).__init__()

        self.K = 5

        self.cb1_1 = Conv_block3D(ch_in,ch_step)
        self.up1 = Up_Bn3D(ch_step)
        self.shift1 = Shift3D()

        self.cb2_1 = Conv_block3D(ch_in,ch_step)
        self.cb2_2 = Conv_block3D(2*ch_step,2*ch_step)
        self.up2 = Up_Bn3D(2*ch_step)
        self.shift2 = Shift3D()

        self.cb3_1 = Conv_block3D(ch_in,ch_step)
        self.cb3_2 = Conv_block3D(3*ch_step,3*ch_step)
        self.up3 = Up_Bn3D(3*ch_step)
        self.shift3 = Shift3D()

        self.cb4_1 = Conv_block3D(ch_in,ch_step)
        self.cb4_2 = Conv_block3D(4*ch_step,4*ch_step)
        self.up4 = Up_Bn3D(4*ch_step)
        self.shift4 = Shift3D()

        self.cb5_1 = Conv_block3D(ch_in,ch_step)
        self.cb5_2 = Conv_block3D(5*ch_step,5*ch_step)
        self.up5 = Up_Bn3D(5*ch_step)
        self.shift5 = Shift3D()

        self.cb6_1 = Conv_block3D(ch_in,ch_step)
        self.cb6_2 = Conv_block3D(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv3d(6*ch_step, 3, 1, padding=0, bias=True)

    def forward(self, z):
        K = self.K

        t = 2**torch.arange(K-1,-1,-1,out=torch.FloatTensor())
        q0_h = self.n_h%32
        q0_w = self.n_w%32
        q0_d = self.n_d%32
        drop_conf_h = torch.zeros(K)
        drop_conf_w = torch.zeros(K)
        drop_conf_d = torch.zeros(K)
        for i in range(K):
            drop_conf_h[i] = (q0_h - torch.sum(t*drop_conf_h)) >= 2**(K-(i+1))
            drop_conf_w[i] = (q0_w - torch.sum(t*drop_conf_w)) >= 2**(K-(i+1))
            drop_conf_d[i] = (q0_d - torch.sum(t*drop_conf_d)) >= 2**(K-(i+1))
        drop_conf_h = drop_conf_h.int()
        drop_conf_w = drop_conf_w.int()
        drop_conf_d = drop_conf_d.int()

        y = self.cb1_1(z[5])
        y = self.up1(y)
        y = self.shift1(y,[drop_conf_h[0],drop_conf_w[0],drop_conf_d[0]])
        z4_temp = self.cb2_1(z[4])
        s = z4_temp.size()
        y = y[:,:,:s[2],:s[3],:s[4]]
        y = torch.cat((y,z4_temp),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = self.shift2(y,[drop_conf_h[1],drop_conf_w[1],drop_conf_d[1]])
        z3_temp = self.cb3_1(z[3])
        s = z3_temp.size()
        y = y[:,:,:s[2],:s[3],:s[4]]
        y = torch.cat((y,z3_temp),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = self.shift3(y,[drop_conf_h[2],drop_conf_w[2],drop_conf_d[2]])
        z2_temp = self.cb4_1(z[2])
        s = z2_temp.size()
        y = y[:,:,:s[2],:s[3],:s[4]]
        y = torch.cat((y,z2_temp),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = self.shift4(y,[drop_conf_h[3]+1,drop_conf_w[3]+1,drop_conf_d[3]+1])
        z1_temp = self.cb5_1(z[1])
        s = z1_temp.size()
        y = y[:,:,:s[2],:s[3],:s[4]]
        y = torch.cat((y,z1_temp),1)
        y = self.cb5_2(y)
        y = self.up5(y)
        y = self.shift5(y,[drop_conf_h[4],drop_conf_w[4],drop_conf_d[4]])
        z0_temp = self.cb6_1(z[0])
        s = z0_temp.size()
        y = y[:,:,:s[2],:s[3],:s[4]]
        y = torch.cat((y,z0_temp),1)
        y = self.cb6_2(y)
        y = self.last_conv(y)
        return y

cellseed = cp.ElementwiseKernel(
    'uint64 scale, uint64 x, uint64 y, uint64 z, uint64 ch',
    'uint64 seed',
    'const unsigned int spatial_period = 65536u; \
    const unsigned int scale_period = 5u; \
    const unsigned int ch_period = 3u; \
    seed = (scale %  scale_period) * scale_period * ch_period \
    * spatial_period * spatial_period \
    + (ch %  ch_period) * ch_period * spatial_period *spatial_period\
    + (z %  spatial_period) * spatial_period * spatial_period  \
    + (y %  spatial_period) * spatial_period \
    + (x %  spatial_period); \
    if (seed == 0u) seed = 1u;',
    'cellseed')

wang_hash64 = cp.ElementwiseKernel(
    'uint64 key',
    'uint64 out_key',
    'key = (~key) + (key << 21); \
    key = key ^ (key >> 24); \
    key = (key + (key << 3)) + (key << 8);\
    key = key ^ (key >> 14); \
    key = (key + (key << 2)) + (key << 4); \
    key = key ^ (key >> 28); \
    out_key = key + (key << 31);',
    'wang_hash64')

xorshift64star = cp.ElementwiseKernel(
    'uint64 x',
    'float32 r',
    'x ^= x >> 12; \
    x ^= x << 25; \
    x ^= x >> 27; \
    x = x * 0x2545F4914F6CDD1D; \
    r = ((float) x) / ((float) 18446744073709551615u)',
    'xorshift64star')

def sample_noise(x1,x2,y1,y2,z1,z2,n_ch,scale):
    x_size = x2-x1
    y_size = y2-y1
    z_size = z2-z1
    total_size = x_size*y_size*z_size*n_ch

    scale_array = scale*cp.ones((total_size),dtype=np.uint64)

    x_array = cp.arange(x1,x2,dtype=np.uint64)
    x_array = cp.expand_dims(x_array, axis=0)
    x_array = cp.expand_dims(x_array, axis=2)
    x_array = cp.expand_dims(x_array, axis=3)
    x_array = cp.tile(x_array, (n_ch,1,y_size,z_size))
    x_array = x_array.reshape((total_size))

    y_array = cp.arange(y1,y2,dtype=np.uint64)
    y_array = cp.expand_dims(y_array, axis=0)
    y_array = cp.expand_dims(y_array, axis=1)
    y_array = cp.expand_dims(y_array, axis=3)
    y_array = cp.tile(y_array, (n_ch,x_size,1,z_size))
    y_array = y_array.reshape((total_size))

    z_array = cp.arange(z1,z2,dtype=np.uint64)
    z_array = cp.expand_dims(z_array, axis=0)
    z_array = cp.expand_dims(z_array, axis=1)
    z_array = cp.expand_dims(z_array, axis=2)
    z_array = cp.tile(z_array, (n_ch,x_size,y_size,1))
    z_array = z_array.reshape((total_size))

    ch_array = cp.arange(n_ch,dtype=np.uint64)
    ch_array = cp.expand_dims(ch_array, axis=1)
    ch_array = cp.expand_dims(ch_array, axis=2)
    ch_array = cp.expand_dims(ch_array, axis=3)
    ch_array = cp.tile(ch_array, (1,x_size,y_size,z_size))
    ch_array = ch_array.reshape((total_size))

    seed = cellseed(scale_array, x_array, y_array, z_array, ch_array)
    hash_seed = wang_hash64(seed)
    noise = xorshift64star(hash_seed)
    noise_cpu = cp.asnumpy(noise)
    noise_cpu = noise_cpu.reshape((n_ch,x_size,y_size,z_size))

    return noise_cpu


model_folder = 'Trained/2020-01-10_brown016_exemplar_3D_2036'

#load model
n_input_ch = 3
generator = MultiScaleGen3D(ch_in=n_input_ch,  ch_step=4)
generator.load_state_dict(torch.load('./' + model_folder + '/params.pytorch'))
generator.cuda()
# generator.eval()



slice = 1 # save middle slices as images

ref = [128,128,128] # reference coordinates


# the sample can be generated by pieces of this size
piece_height = 64
piece_width = 64
piece_depth = 64

# total size of the sample
total_H = 256
total_W =  256
total_D =  256

full_vol = np.zeros((3,total_H,total_W,total_D),dtype='uint8')
imagenet_mean = torch.Tensor([0.40760392, 0.45795686, 0.48501961])
imagenet_mean = imagenet_mean.repeat(piece_height, piece_width, piece_depth, 1).permute(3,0,1,2).cuda()
h = piece_height
w = piece_width
d = piece_depth
n_blocks = 0
for i in range(0,total_H,h):
    for j in range(0,total_W,w):
        for k in range(0,total_D,d):
            print(str(i) + ',' + str(j) + ',' + str(k))
            d1_1 = ref[0] + i
            d2_1 = ref[1] + j
            d3_1 = ref[2] + k
            d1_2 = d1_1 + h
            d2_2 = d2_1 + w
            d3_2 = d3_1 + d

            generator.n_h = d1_1
            generator.n_w = d2_1
            generator.n_d = d3_1


            z_0 = torch.from_numpy(sample_noise(d1_1-4,d1_2+4,d2_1-4,d2_2+4,d3_1-4,d3_2+4,3,0)).unsqueeze(0)
            z_1 = torch.from_numpy(sample_noise(int(math.floor(d1_1/2))-5,int(math.floor(d1_1/2))+int(math.ceil(h/2))+5,int(math.floor(d2_1/2))-5, \
                int(math.floor(d2_1/2))+int(math.ceil(w/2))+5,int(math.floor(d3_1/2))-5,int(math.floor(d3_1/2))+int(math.ceil(d/2))+5,3,1)).unsqueeze(0)
            z_2 = torch.from_numpy(sample_noise(int(math.floor(d1_1/4))-6,int(math.floor(d1_1/4))+int(math.ceil(h/4))+6,int(math.floor(d2_1/4))-6, \
                int(math.floor(d2_1/4))+int(math.ceil(w/4))+6,int(math.floor(d3_1/4))-6,int(math.floor(d3_1/4))+int(math.ceil(d/4))+6,3,2)).unsqueeze(0)
            z_3 = torch.from_numpy(sample_noise(int(math.floor(d1_1/8))-6,int(math.floor(d1_1/8))+int(math.ceil(h/8))+6,int(math.floor(d2_1/8))-6, \
                int(math.floor(d2_1/8))+int(math.ceil(w/8))+6,int(math.floor(d3_1/8))-6,int(math.floor(d3_1/8))+int(math.ceil(d/8))+6,3,3)).unsqueeze(0)
            z_4 = torch.from_numpy(sample_noise(int(math.floor(d1_1/16))-6,int(math.floor(d1_1/16))+int(math.ceil(h/16))+6,int(math.floor(d2_1/16))-6, \
                int(math.floor(d2_1/16))+int(math.ceil(w/16))+6,int(math.floor(d3_1/16))-6,int(math.floor(d3_1/16))+int(math.ceil(d/16))+6,3,4)).unsqueeze(0)
            z_5 = torch.from_numpy(sample_noise(int(math.floor(d1_1/32))-4,int(math.floor(d1_1/32))+int(math.ceil(h/32))+4,int(math.floor(d2_1/32))-4, \
                int(math.floor(d2_1/32))+int(math.ceil(w/32))+4,int(math.floor(d3_1/32))-4,int(math.floor(d3_1/32))+int(math.ceil(d/32))+4,3,5)).unsqueeze(0)

            z_list = [z_0,z_1,z_2,z_3,z_4,z_5]
            z_var = [Variable(z, volatile=True).cuda() for z in  z_list]

            sample = generator(z_var)
            out = sample.data[0,:,:,:,:].squeeze(0)*(1./255) + imagenet_mean
            out = 255*out.clamp(0,1)

            full_vol[:,i:i+h,j:j+w,k:k+d] = np.uint8(out.cpu().numpy())
            n_blocks = n_blocks + 1
            del z_0, z_1, z_2, z_3, z_4, z_5


print(str(n_blocks) + ' blocks of size ' + str(piece_height) + 'x' + str(piece_width) + 'x' + str(piece_depth))
# save solid texture in BGR (color channels)
np.save('./' + model_folder + '/offline_ondemand_volume_' + str(total_H) + '_' + str(total_W) + '_' + str(total_D), full_vol)

# switch to RGB before slicing and saving images
full_vol = full_vol[[2,1,0],:,:,:]
if slice:
    middle_slice = full_vol[:,int(total_H/2),:,:].squeeze().transpose(1,2,0)
    im = Image.fromarray(middle_slice)
    im.save('./' + model_folder + '/offline_ondemand_volume_' + str(total_H) + '_' + str(total_W) + '_' + str(total_D) + '_middle_slice1.jpg')

    middle_slice = full_vol[:,:,int(total_W/2),:].squeeze().transpose(1,2,0)
    im = Image.fromarray(middle_slice)
    im.save('./' + model_folder + '/offline_ondemand_volume_' + str(total_H) + '_' + str(total_W) + '_' + str(total_D) + '_middle_slice2.jpg')

    middle_slice = full_vol[:,:,:,int(total_D/2)].squeeze().transpose(1,2,0)
    im = Image.fromarray(middle_slice)
    im.save('./' + model_folder + '/offline_ondemand_volume_' + str(total_H) + '_' + str(total_W) + '_' + str(total_D) + '_middle_slice3.jpg')
