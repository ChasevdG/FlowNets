#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import os
import sys
import time
import argparse
import datetime
import math
import pickle


#import cv2

#from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from utils.autoaugment import CIFAR10Policy

import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


# In[2]:


# from utils.BBBlayers import GaussianVariationalInference
# from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.FlowTest import FlowTestNet
from utils.BayesianModels.FlowResNet import resnet18 as BBBCNN
from utils.NonBayesianModels.Resnet import resnet18 as CNN

# from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet
#from utils.BayesianModels.BayesianExperimentalCNNModel import BBBCNN1

# In[3]:


net_type = 'flowtest'
dataset = 'Resnet18-CIFAR10-SGD-Fixed'
outputs = 10
inputs = 3
resume = False
num_epochs = 200
lr = 0.2
wd = 0#5e-4
nest = False
num_samples = 1
beta_type = "Standard"
dropout = False
opt = "SGD"
print(dataset)

# In[4]:


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.set_device(0)
else:
    print('Not Using Cuda')

# In[5]:


# number of subprocesses to use for data loading
num_workers = 5
# how many samples per batch to load
batch_size = 256
# percentage of training set to use as validation
valid_size = 0.2


# In[6]:


# convert data to a normalized torch.FloatTensor
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transform)





# In[7]:


# obtain training indices that will be used for validation

print("Size of image")
print(test_data[0][0].size())

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


# In[8]:


# define samplers for obtaining training and validation batches
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)


# In[9]:


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True,num_workers=2)
#valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
#    sampler=valid_sampler, shuffle = True,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    shuffle = False,num_workers=2)


# In[10]:


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# In[11]:


#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# helper function to un-normalize and display an image
#def imshow(img):
    # Uncomment if normalizing the data
    #img = img / 2 + 0.5  # unnormalize
#    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# In[12]:


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display
#print(image)
#cv2.imshow("window", image)
       #cv2.waitKey(0)

       #tf = transform.Resize((10,10))
       #resized = tf(image)
       #print(resized)
       #np.set_printoptions(threshold=sys.maxsize)
       #print(images)
       # plot the images in the batch, along with the corresponding labels
       #fig = plt.figure(figsize=(25, 4))
       # display 20 images
       #for idx in np.arange(20):
       #    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
       #    imshow(images[idx])
       #    ax.set_title(classes[labels[idx]])


# In[13]:


rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

       # Architecture
if (net_type == 'lenet'):
    net = BBBLeNet(outputs,inputs)
elif (net_type == 'alexnet'):
    net = BBBAlexNet( outputs, inputs)
elif (net_type == '3conv3fc'):
    net = BBB3Conv3FC(outputs,inputs)
elif (net_type == 'BBBResnet'):
    net = BBBCNN(inputs)
elif (net_type == 'NBResnet'):
    net = CNN(inputs)
elif (net_type == 'VDOResnet'):
    net = CNN(inputs, dropout_net = True)
elif (net_type == 'flowtest'):
    net = FlowTestNet(outputs,inputs)
else:
    print('Error : Network should be either [LeNet / AlexNet / 3Conv3FC / Resnet]')


       # In[15]:


if use_cuda:
    net.cuda()


       # In[16]:


ckpt_name = f'model_{net_type}_{dataset}_bayesian.pt'


       # In[17]:


criterion = nn.CrossEntropyLoss()
def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1) 
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m 
    else:
        print('Error : Network should be either [Blundell / Soenderby / Blundell')
    return beta


# In[18]:


def elbo(out, y, kl, beta):
    loss = F.cross_entropy(out, y)
    return loss + beta * kl


       # In[19]:


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if(use_cuda):inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        if hasattr(net, 'probforward') and callable(net.probforward):
               outputs, kl = net.probforward(inputs)
               loss = elbo(outputs, targets, kl, get_beta(epoch, len(train_data), beta_type))
               loss.backward()
               optimizer.step()
               pred = torch.max(outputs, dim=1)[1]
               correct += torch.sum(pred.eq(targets)).item()
               total += targets.numel()
        else:
               outputs = net.forward(inputs)
               loss = criterion(outputs, targets)
               loss.backward()
               optimizer.step()
               pred = torch.max(outputs, dim=1)[1]
               correct += torch.sum(pred.eq(targets)).item()
               total += targets.numel()

    print(f'[TRAIN] Acc: {100.*correct/total:.3f}')
    sys.stdout.flush()
       
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    accuracy_max = 0    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if hasattr(net, 'probforward') and callable(net.probforward):
               if(use_cuda):
                  inputs, targets = inputs.cuda(), targets.cuda()
               outputs, _ = net.probforward(inputs)
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
               accuracy = 100.*correct/total
            else:
               if(use_cuda):
                   inputs, targets = inputs.cuda(), targets.cuda()
               outputs = net.forward(inputs)
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
               accuracy = 100.*correct/total
        print(f'[TEST] Acc: {accuracy:.3f}')
        

    torch.save(net.state_dict(), ckpt_name)
    


# In[21]:


epochs = [80, 60, 40, 20]
count = 0


# In[22]:


from torch.optim import Adam
from torch.optim import SGD
from torch.optim import Adamax

for epoch in epochs:
    if opt == "Adam":
        optimizer = Adam(net.parameters(), lr=lr,weight_decay=wd)
    elif opt=="Adamax":
        optimizer = Adamax(net.parameters(),lr=lr)
    elif opt=="SGD":
        optimizer = SGD(net.parameters(),lr=lr,momentum=.9,weight_decay = wd,nesterov=nest )

    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    lr /= 10


# In[23]:


# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if use_cuda:
    images = images.cuda()

# get sample outputs
output, kl = net.probforward(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
#fig = plt.figure(figsize=(25, 4))
#for idx in np.arange(20):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    imshow(images.cpu()[idx])
#    ax.set_title("{} {})".format(classes[preds[idx]], classes[labels[idx]]),
#                 color=("green" if preds[idx]==labels[idx].item() else "red"))

