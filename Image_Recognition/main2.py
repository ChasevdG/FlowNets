
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
import sys

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
from utils.BayesianModels.FlowAlexNet import BBBAlexNet 
# from utils.BayesianModels.BayesianLeNet import BBBLeNet
# from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet


# In[3]:


net_type = 'alexnet'
dataset = 'CIFAR10'
outputs = 10
inputs = 3
resume = False
n_epochs = 150
lr = 0.005
weight_decay = 0.000
num_samples = 1
beta_type = "none"
#beta_type = "Standard"
resize=32


# In[4]:


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


# In[5]:


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 256
# percentage of training set to use as validation
valid_size = 0.2


# In[6]:


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transform)


# In[7]:


# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


# In[8]:


# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# In[9]:


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


# In[10]:


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# helper function to un-normalize and display an image
    # Uncomment if normalizing the data
    #img = img / 2 + 0.5  # unnormalize

iter(train_loader)

# Architecture
if (net_type == 'lenet'):
    net = BBBLeNet(outputs,inputs)
elif (net_type == 'alexnet'):
    net = BBBAlexNet( outputs, inputs)
    #net = BBBAlexNet(outputs,inputs)
elif (net_type == '3conv3fc'):
    net = BBB3Conv3FC(outputs,inputs)
elif (net_type == 'flow'):
    net = FlowTestNet(outputs,inputs)
else:
    print('Error : Network should be either [LeNet / AlexNet / 3Conv3FC')

torch.autograd.set_detect_anomaly(True)
# In[15]:


if use_cuda:
    net.cuda()


# In[16]:


ckpt_name = f'model_{net_type}_{dataset}_bayesian.pt'
ckpt_name


# In[17]:



def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1) 
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
        #print(m) 
    else:
        beta = 0
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
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, kl = net.probforward(inputs)
        #norm_factor = torch.sum(outputs,1)
        #outputs = torch.t(torch.t(outputs)/norm_factor)
        loss = elbo(outputs, targets, kl, get_beta(epoch, len(train_data), beta_type))
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100.*correct/total:.3f} Loss: {loss:.3f}')
    sys.stdout.flush()
    return loss

# In[20]:


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n=1
    accuracy_max = 0    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if n == 1:
               outputs, kl = net.probforward(inputs,10000)
               n=0
            else:
               outputs, kl = net.probforward(inputs)

            #norm_factor = torch.sum(outputs,1)
            #outputs = torch.t(torch.t(outputs)/norm_factor)

            loss = elbo(outputs, targets, kl, get_beta(epoch, len(train_data), beta_type))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total
        print(f'[TEST] Acc: {accuracy:.3f} Loss: {loss:.3f}')
    torch.save(net.state_dict(), ckpt_name)
    return loss
    


# In[21]:


epochs = [80, 60, 40, 20]
count = 0
tr_losses = np.array([])
te_losses = np.array([])
# In[22]:


from torch.optim import Adam
for epoch in epochs:
    optimizer = Adam(net.parameters(), lr=lr)
    for _ in range(epoch):
        losses = train(count)
        with torch.no_grad():
            tr_losses = np.append(tr_losses,losses.cpu().detach())
        losses = test(count)
        with torch.no_grad():
            te_losses = np.append(te_losses,losses.cpu().detach())
        count += 1
        np.savetxt('Train_Losses.txt',tr_losses)
        np.savetxt('Test_Losses.txt',te_losses)
    lr /= 10
# In[23]:


# obtain one batch of test images
#dataiter = iter(test_loader)
#images, labels = dataiter.next()
#images.numpy()

# move model inputs to cuda, if GPU available
#if use_cuda:
#    images = images.cuda()

# get sample outputs
#output, kl = net.probforward(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))

