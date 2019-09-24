import torch.nn as nn
# from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
# from utils.conv2d import BBBConv2d
# from utils.linear import BBBLinearFactorial
import math
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial


class FlowTestNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(FlowTestNet, self).__init__()
        flow = False
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)
        n = 3072 
        self.classifier1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , n,flow=flow)
        self.classifier2 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , n,flow=flow)
        self.classifier3 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , n,flow=flow)
        self.classifier4 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , n,flow=flow)
        self.classifier5 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , n,flow=flow)
 
        self.classifier6 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init,n , outputs,flow=flow)
        self.flatten_layer = Flatten()


        # self.flatten = FlattenLayer(1 * 1 * 128)
        # self.fc1 = BBBLinearFactorial(q_logvar_init, N, p_logvar_init, 1* 1 * 128, outputs)


        layers = [self.flatten_layer,self.classifier1,self.classifier2,self.classifier3,self.classifier4,self.classifier5,self.classifier6]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'probforward') and callable(layer.probforward):
                x, _kl, = layer.probforward(x)
            else:
                x = layer.forward(x)
        x = x.view(x.size(0), -1)
        #x, _kl = self.classifier.probforward(x)
        kl += _kl
        logits = x
        return logits, kl


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    # def probforward(self, x):
    #     'Forward pass with Bayesian weights'
    #     kl = 0
    #     for layer in self.layers:
    #         if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
    #             x, _kl, = layer.convprobforward(x)
    #             kl += _kl

    #         elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
    #             x, _kl, = layer.fcprobforward(x)
    #             kl += _kl
    #         else:
    #             x = layer(x)
    #     logits = x
    #     return logits, kl
