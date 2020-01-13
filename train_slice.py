# train_slice.py
#
# Trains the generator network given a set of examples
#
# Code for the article:
# On Demand Solid Texture Synthesis Using Deep 3D Networks
# J. Gutierrez, J. Rabin, B. Galerne, T. Hurtut
#
# Author: Jorge Gutierrez
# Last modified: 08 Jan 2020
# Based on https://github.com/leongatys/PytorchNeuralStyleTransfer
#


import sys
import random
import datetime
import os
import numpy
import math
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import torchvision
from torchvision import transforms

try:
    import display
except ImportError:
    print('Not displaying')
    pass

if 'display' not in sys.modules:
    disp = 0
else:
    disp = 1


# deterministic training
# torch.backends.cudnn.enabled = False
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

# vgg definition from https://github.com/leongatys
class VGG(nn.Module):
    def __init__(self, pool='max', pad=1 ):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=pad)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

# generator's 3D convolutional blocks
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

# shift compensation for on demand evaluation
class Shift3D(nn.Module):
    def __init__(self):
        super(Shift3D, self).__init__()

    def forward(self, x, shift):
        x = torch.cat((x[:,:,shift[0]:,:,:],x[:,:,:shift[0],:,:]),2)
        x = torch.cat((x[:,:,:,shift[1]:,:],x[:,:,:,:shift[1],:]),3)
        x = torch.cat((x[:,:,:,:,shift[2]:],x[:,:,:,:,:shift[2]]),4)
        return x

# generator
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

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w*c)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

# pre and post processing for images
prep = transforms.Compose([
        transforms.ToTensor(),
        #turn to BGR
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        #subtract imagenet mean
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
        ])

postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        #add imagenet mean
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                            std=[1,1,1]),
        #turn to RGB
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
        ])

postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

# dependency values (specific to the architecture of MultiScaleGen3D)
def dep_values(scale):
    if scale == 0:
        ck = 4
    elif scale == 5:
        ck = math.ceil((dep_values(scale-1)-2)/2) + 2
    else:
        ck = math.ceil((dep_values(scale-1)-2)/2) + 4
    return ck


# size of noise input at a given scale to generate a volume of size hxwxd
def size_input(h,w,d,scale):
    s = [math.ceil(h/(math.pow(2, scale))+2*dep_values(scale)),
         math.ceil(w/(math.pow(2, scale))+2*dep_values(scale)),
         math.ceil(d/(math.pow(2, scale))+2*dep_values(scale))]
    return s


# create generator network
n_input_ch = 3
gen = MultiScaleGen3D(ch_in=3, ch_step=4)
params = list(gen.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.data.numpy().size
print('Generator''s total number of parameters = ' + str(total_parameters))

# get descriptor network
vgg = VGG(pool='avg', pad=1)
vgg.load_state_dict(torch.load('./Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.cuda()


# These lists must have the same number of elements (1-3)
inputs_names = ['brown016_exemplar.png','brown016_exemplar.png','brown016_exemplar.png']
directions = [0,1,2]

# test folder, backup and results
time_info = datetime.datetime.now()
out_folder_name = time_info.strftime("%Y-%m-%d") + '_' \
                  + inputs_names[0][:-4] \
                  + '_3D' + time_info.strftime("_%H%M")
if not os.path.exists('./Trained/' + out_folder_name):
    os.mkdir( './Trained/' + out_folder_name)

# load images
input_textures = [Image.open('./Textures/' + name) for name in inputs_names]
input_textures_torch = [Variable(prep(img)).unsqueeze(0).cuda()
                        for img in input_textures]
# display images
if disp:
    for i,img in enumerate(input_textures):
        img_disp = numpy.asarray(img, dtype="int32")
        display.image(img_disp, win=['input'+str(i)],
                    title=['Input texture d'+str(i)])

#define layers, loss functions, weights and compute optimization target
loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
w = [1,1,1,1,1]


#compute optimization targets
targets = []
for img in input_textures_torch:
    targets.append([GramMatrix()(f).detach() for f in vgg(img, loss_layers)])

# training parameters
slice_size = 128 # training slice resolution (best if same as examples)
iterations = 3000
show_iter = 10 # display every show_iter iterations
save_params = 1000 # save parameters every save_params iterations
save_slice = 100 # save generated slice every save_slice iterations
learning_rate = 0.1
batch_size = 8

# reference coordinate of the slice initialization
slice_coords = [0,0,0]

gen.cuda()
optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
loss_history = numpy.zeros(iterations)
# run training
for n_iter in range(iterations):
    optimizer.zero_grad()

    #Multiview loss
    for idx,d in enumerate(directions):
        target = targets[idx]
        # slice dimensions - 1 voxel thick in dimension d
        v_sizes = [slice_size for N in range(3)]
        v_sizes[d] = 1
        # reference coordinate of the slice reset
        slice_coords = [0,0,0]

        # batch evaluating one sample at a time to save memory for resolution
        for i in range(batch_size):
            # stochastic selection of one slice (determines shift config)
            slice_coords[d] = random.randint(0,31)
            gen.n_h = slice_coords[0]
            gen.n_w = slice_coords[1]
            gen.n_d = slice_coords[2]

            # generation of noise inputs
            sz = [size_input(v_sizes[0],v_sizes[1],v_sizes[2],k) for k in range(6)]
            zk = [torch.rand(1,n_input_ch,szk[0],szk[1],szk[2]) for szk in sz]
            z_samples = [Variable(z.cuda()) for z in zk ]

            # slice synthesis
            batch_sample = gen(z_samples)
            if d == 0:
                sample = batch_sample[:,:,0,0:v_sizes[1]:,0:v_sizes[2]]
            if d == 1:
                sample = batch_sample[:,:,0:v_sizes[0],0,0:v_sizes[2]]
            if d == 2:
                sample = batch_sample[:,:,0:v_sizes[0],0:v_sizes[1],0]
            sample = sample.squeeze().unsqueeze(0)

            # loss evaluation
            out = vgg(sample, loss_layers)
            losses = [w[a]*loss_fns[a](f, target[a]) for a,f in enumerate(out)]
            single_loss = (1/(batch_size*len(targets)))*sum(losses)
            single_loss.backward(retain_graph=False)

            loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
            del out, losses, single_loss
            del batch_sample, z_samples, zk

        if disp:
            if n_iter%show_iter == (show_iter-1):
                out_img = postp(sample.data.cpu().squeeze())
                out_img_array = numpy.asarray( out_img, dtype="int32" )
                display.image(out_img_array, win='sample'+str(d),
                              title='Generated sample d'+str(d))

        if n_iter%save_slice == (save_slice-1):
            out_img = postp(sample.data.cpu().squeeze())
            out_img.save('./Trained/' + out_folder_name + '/training_d'
                         + str(d) + '_' + str(n_iter+1) + '.jpg', "JPEG")
        del sample


    print('Iteration: %d, loss: %f'%(n_iter, loss_history[n_iter]))

    if n_iter%save_params == (save_params-1):
        torch.save(gen, './Trained/' + out_folder_name
                   + '/trained_model_' + str(n_iter+1) + '.py')
        torch.save(gen.state_dict(), './Trained/' + out_folder_name
                   + '/params' + str(n_iter+1) + '.pytorch')

    optimizer.step()


# save final model and training history
torch.save(gen,'./Trained/'+out_folder_name +'/trained_model.py')
torch.save(gen.state_dict(),'./Trained/'+out_folder_name+'/params.pytorch')
numpy.save('./Trained/'+out_folder_name+'/loss_history',loss_history)
