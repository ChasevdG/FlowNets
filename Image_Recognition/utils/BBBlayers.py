import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import warnings


class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

class BBBConv2d(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,flow=False):
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
        self.flow = flow
        if(self.flow):
           self.flow_w = Parameter(torch.Tensor(out_features, in_features))
           self.flow_b = Parameter(torch.Tensor(out_features, in_features))
           self.flow_u = Parameter(torch.Tensor(out_features, in_features))
           self.flow_h = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)
        if(self.flow):
           self.flow_w.data.uniform_(-0.01, 0.01)
           self.flow_u.data.uniform_(-0.01, 0.01)
           self.flow_b.data.uniform_(-0.01, 0.01)
           print("Using Flow")

    def forward(self, input):
        warnings.warn("Using non-probabilistic forward on BBB layer!", UserWarning)
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        #print(input.size())
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out


    def probforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
       	#print(input.size()) 
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_.sum()
        if(self.flow):
           z = weight
           w = self.flow_w
           b = self.flow_b
           h = self.flow_h
           u = self.flow_u
           #print(z,w,b,u,h)
           weight, det = self.planar_flow(z,w,b,h,u)
           kl = kl + sum(det) 
        return out, kl

class BBBLinearFactorial(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_features, out_features, bias=False, flow = False, flow_function_type = None,flow_length = 16):
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

    def probforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        kl = kl_.sum()
        if(self.flow):
           for transform in self.flow_layers:
              z = weight
              z_prime, det = transform(z)
              weight = z_prime/det
              kl = kl + sum(det)
        out = F.linear(input, weight, bias)
        return out, kl
 
    def flow_Function(flow_type):
       if(flow_type == "planar"):
           fun = NormalizingFlow.planar, NormalizingFlow.planar_det
       else:
           fun = None
       return fun
class FlowLayer(nn.Module):
    def __init__(self, out_features, in_features):

        super(FlowLayer, self).__init__()

        #Bayes-by-Backprop params
        self.w = Parameter(torch.Tensor(out_features, in_features))
        self.b = Parameter(torch.Tensor(out_features, in_features))
        self.u = Parameter(torch.Tensor(out_features, in_features))
        self.h = nn.Tanh()
           #self.flow_transform, self.flow_det = flow_Function(flow_function_type)

        self.reset_parameters()
    def reset_parameters(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self,z):
        det = 1 + self.u*(1-self.h(self.w*z+self.b)**2)
        z_new = z + self.u*self.h(self.w*z+self.b)
        return z_new, det.abs() 



