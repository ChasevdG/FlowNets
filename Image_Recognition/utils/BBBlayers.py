import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt

class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

class BBBConv2d(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,flow=False, flow_length=1):
        super(BBBConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.q_logvar_init = q_logvar_init
        self.p_logvar_init = p_logvar_init
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.flow = False#flow
        if(self.flow):
           self.flow_layers = nn.Sequential(*(FlowLayer(out_channels, in_channels,kernel_size,kernel_size) for _ in range(flow_length)))
           print("Using Flow")

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)

    def forward(self, input):
        warnings.warn("Using non-probabilistic forward on BBB layer!", UserWarning)
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        if(self.flow):
           for transform in self.flow_layers:
              z = weight
              z_prime, det = transform(z)
              weight = z_prime
              kl_ = kl_
        #print(input.size())
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out


    def probforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = -0.5*(self.q_logvar_init + (weight - self.mu_weight)*(weight - self.mu_weight)*sig_weight.reciprocal())
        #kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2)  - 0.5
        bias = None

        if(self.flow):
           for transform in self.flow_layers:
              z = weight
              z_prime, det = transform(z)
              #print("DETERM:1",det)
              weight = z_prime*det

              kl_ = kl_ + det

       	#print(input.size()) 
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_.sum()
        #print(kl)
        return out, kl


class BBBLinearFactorial(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_features, out_features, bias=False, flow = False, flow_function_type = None,flow_length = 3):
        super(BBBLinearFactorial, self).__init__()
        
        #Bayes-by-Backprop params
        self.q_logvar_init = q_logvar_init
        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.flow = flow
        #Normalizing Flow Params
        if(self.flow):
           self.flow_layers = nn.Sequential(*(FlowLayer(out_features, in_features) for _ in range(flow_length)))
	   #self.flow_transform, self.flow_det = flow_Function(flow_function_type)


        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)
        self.eps_weight.data.zero_()

    def forward(self, input):
        raise NotImplementedError()
        
    def planar_flow(self,z,w,b,h,u):
        det = 1 + u*(1-h(w*z+b)**2)
        z_new = z + u*h(w*z+b)
        return z_new, det.abs()

    def probforward(self, input, num_samples =1):
        out, loss, flow_weights, flow_weight_0 = self.pbforward(input)
        output_mat = np.array([])
        with torch.no_grad():
                a = flow_weights.cpu()
                output_mat = np.array(a[0].tolist())
                n = len(output_mat)
                output_mat = output_mat.reshape(len(output_mat),1)

                b = flow_weight_0.cpu()
                input_mat = np.array(b[0].tolist())
                n = len(input_mat)
                input_mat = input_mat.reshape(len(output_mat),1)
        for _ in range(num_samples-1):
            cur_out, cur_loss,weight,weight_0 = self.pbforward(input)
            out += cur_out
            with torch.no_grad():
                a = weight.cpu()
                #For Cov after flow
                temp = np.array(a[0])
                temp = temp.reshape(n,1)
                output_mat = np.concatenate((output_mat,temp), axis = 1)
                #Cov before the flow
                b = weight_0.cpu()
                temp = np.array(b[0])
                temp = temp.reshape(n,1)
                input_mat = np.concatenate((input_mat,temp), axis = 1)

            loss += cur_loss
        if(num_samples > 1):
            X = np.corrcoef(input_mat)
            Y = np.corrcoef(output_mat)
            #print(X.shape)
            
            np.save('in_cov2',X)
            np.save('out_cov2',Y)
            #np.savetxt('in_mat.txt',input_mat)
            #np.savetxt('out_mat.txt',output_mat)
            #print("Saved cov")
        #print(output_mat.shape)
        return out/num_samples, loss/num_samples
            



    def pbforward(self,input):
        sig_weight = torch.log(1+torch.exp(self.sigma_weight))
        weight_0 = self.mu_weight + sig_weight * self.eps_weight.normal_()
        q0_z0 = -0.5*(torch.log(sig_weight) + (weight_0 - self.mu_weight)*(weight_0 - self.mu_weight)*sig_weight.reciprocal()*sig_weight.reciprocal()) #log q_0(z_0)
        bias = None
        sum_det = 0
        weight = weight_0

        if(self.flow):
           for transform in self.flow_layers:
              z = weight
              z_prime, det = transform(torch.t(z))
              weight = z_prime
              sys.stdout.flush()
              sum_det = det.sum()
              weight = torch.t(weight)
           p_z = -.5*weight*weight
           loss = torch.sum(q0_z0 - p_z) - sum_det
        else: 
             kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2)  - 0.5
             loss = kl_.sum()
        out = F.linear(input, weight, bias)
        return out, loss, weight, weight_0
 
def check_corr(model, num_samples):
       
         mat = np.array([])
         for _ in range(num_samples):
            sig_weight1 = torch.exp(model.sigma_weight)
            weight_01 = model.mu_weight + sig_weight1 * model.eps_weight.normal_()
            weight1 = weight_01
            for transform in self.flow_layers:
              z1 = weight1
              z1_prime, _ = transform(torch.t(z1))
              weight1 = z1_prime
              weight1 = torch.t(weight1)
            np.append(mat,weight1.cpu())
         print(mat.shape)
         return mat

class FlowLayer(nn.Module):
    def __init__(self,out_features, in_features, conv_dim1=0,conv_dim2=0):

        super(FlowLayer, self).__init__()
        if(conv_dim1==0 & conv_dim2==0):
            self.w = Parameter(torch.Tensor(in_features,1))
            self.b = Parameter(torch.Tensor(1))
            self.u = Parameter(torch.Tensor(in_features,1))
        else:
            self.w = Parameter(torch.Tensor(in_features,conv_dim1,conv_dim2))
            self.b = Parameter(torch.Tensor(in_features,conv_dim1,conv_dim2))
            self.u = Parameter(torch.Tensor(in_features,conv_dim1,conv_dim2))
        self.h = nn.Tanh()
        self.reset_parameters()

    def finite_log(self,x):
        return torch.log(x + 1e-7)
    def reset_parameters(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self,z):
        activation = torch.mm(torch.t(self.w),z)
        activation = activation + self.b
        det = 1 + torch.mm(torch.t(self.u),self.w*(1-self.h(activation)**2))
        temp = torch.mm(self.u,self.h(activation))
        z_new = z + temp
        return z_new, self.finite_log(det.abs())


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()
