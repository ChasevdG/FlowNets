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


from utils.BBBlayers import GaussianVariationalInference
from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.BayesianAlexNet import BBBAlexNet
from utils.BayesianModels.BayesianLeNet import BBBLeNet
from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet


# In[3]:


net_type = 'alexnet'
dataset = 'CIFAR10'
outputs = 10
inputs = 3
resume = False
n_epochs = 30
lr = 0.001
weight_decay = 0.0005
num_samples = 1
beta_type = "Blundell"
resize=32


# In[4]:


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


# In[5]:


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.2


# In[6]:


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)


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


# In[ ]:


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


# In[ ]:


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# In[ ]:


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])


# In[ ]:


rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36)) 
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')


# In[ ]:


# Architecture
if (net_type == 'lenet'):
    net = BBBLeNet(outputs,inputs)
elif (net_type == 'alexnet'):
    net = BBBAlexNet(outputs,inputs)
elif (net_type == '3conv3fc'):
        net = BBB3Conv3FC(outputs,inputs)
else:
    print('Error : Network should be either [LeNet / AlexNet / 3Conv3FC')


# In[ ]:


if use_cuda:
    net.cuda()


# In[ ]:


vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


# In[ ]:


ckpt_name = f'model_{net_type}_{dataset}_bayesian.pt'
ckpt_name


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nvalid_loss_min = np.Inf # track change in validation loss\n\nfor epoch in range(1, n_epochs+1):\n\n    # keep track of training and validation loss\n    train_loss = 0.0\n    valid_loss = 0.0\n    \n    m = math.ceil(len(train_data) / batch_size)\n    \n    ###################\n    # train the model #\n    ###################\n    net.train()\n    for batch_idx, (data, target) in enumerate(train_loader):\n        # move tensors to GPU if CUDA is available\n        \n        data = data.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)\n        target = target.repeat(num_samples)\n        if use_cuda:\n            data, target = data.cuda(), target.cuda()\n            \n        if beta_type is "Blundell":\n            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)\n        elif beta_type is "Soenderby":\n            beta = min(epoch / (num_epochs // 4), 1)\n        elif beta_type is "Standard":\n            beta = 1 / m\n        else:\n            beta = 0\n        # clear the gradients of all optimized variables\n        optimizer.zero_grad()\n        # forward pass: compute predicted outputs by passing inputs to the model\n        output,kl = net.probforward(data)\n        # calculate the batch loss\n        loss = vi(output, target, kl, beta)\n        # backward pass: compute gradient of the loss with respect to model parameters\n        loss.backward()\n        # perform a single optimization step (parameter update)\n        optimizer.step()\n        # update training loss\n        train_loss += (loss.item()*data.size(0)) / num_samples\n        \n    ######################    \n    # validate the model #\n    ######################\n    net.eval()\n    for batch_idx, (data, target) in enumerate(valid_loader):\n        data = data.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)\n        target = target.repeat(num_samples)\n        # move tensors to GPU if CUDA is available\n        if use_cuda:\n            data, target = data.cuda(), target.cuda()\n        # forward pass: compute predicted outputs by passing inputs to the model\n        output,kl = net.probforward(data)\n        # calculate the batch loss\n        loss = vi(output, target, kl, beta)\n        # update average validation loss \n        valid_loss += (loss.item()*data.size(0)) / num_samples\n        \n    # calculate average losses\n    train_loss = train_loss/(len(train_loader.dataset) * (1-valid_size))\n    valid_loss = valid_loss/(len(valid_loader.dataset) * valid_size)\n        \n    # print training/validation statistics \n    print(\'Epoch: {} \\tValidation Loss: {:.6f}\'.format(\n        epoch, train_loss, valid_loss))\n    \n    # save model if validation loss has decreased\n    if valid_loss <= valid_loss_min:\n        print(\'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\'.format(\n        valid_loss_min,\n        valid_loss))\n        torch.save(net.state_dict(), ckpt_name)\n        valid_loss_min = valid_loss')


# In[ ]:


net.load_state_dict(torch.load(ckpt_name))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# track test loss\ntest_loss = 0.0\nclass_correct = list(0. for i in range(10))\nclass_total = list(0. for i in range(10))\n\nnet.eval()\nm = math.ceil(len(test_data) / batch_size)\n# iterate over test data\nfor batch_idx, (data, target) in enumerate(test_loader):\n    \n    data = data.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)\n    target = target.repeat(num_samples)\n    # move tensors to GPU if CUDA is available\n    if use_cuda:\n        data, target = data.cuda(), target.cuda()\n    \n    if beta_type is "Blundell":\n        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)\n    elif cf.beta_type is "Soenderby":\n        beta = min(epoch / (cf.num_epochs // 4), 1)\n    elif cf.beta_type is "Standard":\n        beta = 1 / m\n    else:\n        beta = 0\n    # forward pass: compute predicted outputs by passing inputs to the model\n    output, kl = net.probforward(data)\n    # calculate the batch loss\n    loss = vi(output, target, kl, beta)\n    # update test loss \n    test_loss += loss.item()*data.size(0) / num_samples\n    #test_loss += loss.item()\n    # convert output probabilities to predicted class\n    _, pred = torch.max(output, 1) \n    \n    # compare predictions to true label\n    correct_tensor = pred.eq(target.data.view_as(pred))\n    correct = np.squeeze(correct_tensor.numpy()) if not use_cuda else np.squeeze(correct_tensor.cpu().numpy())\n    # calculate test accuracy for each object class\n    for i in range(batch_size):\n        if i >= target.data.shape[0]: # batch_size could be greater than left number of images\n            break\n        label = target.data[i]\n        class_correct[label] += correct[i].item()\n        class_total[label] += 1\n\n# average test loss\ntest_loss = test_loss/len(test_loader.dataset)\nprint(\'Test Loss: {:.6f}\\n\'.format(test_loss))\n\nfor i in range(10):\n    if class_total[i] > 0:\n        print(\'Test Accuracy of %5s: %2d%% (%2d/%2d)\' % (\n            classes[i], 100 * class_correct[i] / class_total[i],\n            np.sum(class_correct[i]), np.sum(class_total[i])))\n    else:\n        print(\'Test Accuracy of %5s: N/A (no training examples)\' % (classes[i]))\n\nprint(\'\\nTest Accuracy (Overall): %2d%% (%2d/%2d)\' % (\n    100. * np.sum(class_correct) / np.sum(class_total),\n    np.sum(class_correct), np.sum(class_total)))')


# In[ ]:


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
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))


# In[ ]:




