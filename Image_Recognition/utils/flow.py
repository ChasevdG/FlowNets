import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class NormalizingFlow(nn.Module):
    def __init__(self,dim,flow_length):
        super().__init__(self, dim, flow_length)

        self.transforms = nn.Sequential(*(
                PlanarFlow(dim) for _ in rangr(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
             PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z):    
        log_jacobians = []
        for tranform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        return z, log_jacobians


def planar(z, w, b, u, h)
    activation = z*w + b
    return z + u * h(activation)


class PlanarTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(1,dim)) # weight
        self.b = nn.Parameter(torch.Tensor(1)) # bias
        self.u = nn.Parameter(torch.Tensor(1,dim)) # scale
        self.h = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):

        self.w.data.uniform_(-0.01,0.01)
        self.u.data.uniform_(-0.01,0.01)
        self.b.data.uniform_(-0.01,0.01)

    def forward(self,z):

        activation = torch.mm(z, self.w.t()) + self.bias
        return z + self.u * self.ReLU(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    def __init__(self, affine):
        super().__init__()

        self.w = affine.w
        self.b = affine.b
        self.u = affine.u
        self.h = affine.h

    def forward(self, z):

        activation = z * self.weight,self.bias)
        psi = (1 - self.h(activation)**2)*self.weight
        det_grad = 1 + torch.mm(psi,self.scale.t())

        return finite_log(det_grad.abs())


#Helper function which prevents 
def finite_log(x):
    return torch.log(z + 1e-7)
