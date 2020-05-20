import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# --- Linear module ---
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
  def __init__(self, in_features, out_features, bias=True):
    super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
    self.weight.fast = None #Lazy hack to add fast weight link
    self.bias.fast = None

  def forward(self, x):
    if self.weight.fast is not None and self.bias.fast is not None:
      out = F.linear(x, self.weight.fast, self.bias.fast)
    else:
      out = super(Linear_fw, self).forward(x)
    return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- Simple ResNet Block ---
class SimpleBlock2(nn.Module):
  maml = False ### ditch maml and just try fearyre 
  def __init__(self, indim, outdim, half_res, leaky=False):
    super(SimpleBlock2, self).__init__()
    self.indim = indim
    self.outdim = outdim
    self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
    self.BN1 = nn.BatchNorm2d(outdim)
    self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
    self.BN2 = FeatureWiseTransformation2d_fw(outdim) # feature-wise transformation at the end of each residual block
    self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

    self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

    self.half_res = half_res

    # if the input number of channels is not equal to the output, then need a 1x1 convolution
    if indim!=outdim:
      self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
      self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)

      self.parametrized_layers.append(self.shortcut)
      self.parametrized_layers.append(self.BNshortcut)
      self.shortcut_type = '1x1'
    else:
      self.shortcut_type = 'identity'

    for layer in self.parametrized_layers:
      init_layer(layer)

  def forward(self, x):
    out = self.C1(x)
    out = self.BN1(out)
    out = self.relu1(out)
    out = self.C2(out)
    out = self.BN2(out)
    short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
    out = out + short_out
    out = self.relu2(out)
    return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

  # --- softplus module ---
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)

        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)

        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# Bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
    self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    self.gamma.requires_grad = False
    self.beta.requires_grad = False
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out

class ResNet2(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out


class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

class ResNet_3(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet_3,self).__init__()
        assert len(list_of_num_layers)==3, 'Can have only 3 stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(3):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

class ResNet_fin(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet_fin,self).__init__()
        assert len(list_of_num_layers)==1, 'Can have only 1 stages'
        trunk = []
        indim = 256
        for i in range(1):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out


def ResNet_fin_func():
    return ResNet_fin(SimpleBlock, [1],[512], True)

def ResNet8(flatten = True):
    flatten = False
    return ResNet_3(SimpleBlock, [1,1,1],[64,128,256], False)

def ResNet10(flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)
def ResNet10_FW(flatten = True):
    return ResNet(SimpleBlock2, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)
def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)
