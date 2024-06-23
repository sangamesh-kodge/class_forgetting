import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import torch 
import random 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/float(stride[0])+1)), int(np.floor((Lin[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/float(stride[1])+1))


def reshape_conv_input_activation(x, conv_layer=None, kernel_size=3, stride=1, padding=0, dilation=1, max_samples=10000):
    ### FAST CODE (Avoid for loops)
    if conv_layer:
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding =  conv_layer.padding 
        dilation = conv_layer.dilation
    if x.shape[-1] > 3*kernel_size[-1]:
        start_index_i =random.randint(0, x.shape[-1]-3*kernel_size[-1])
        start_index_j =random.randint(0, x.shape[-2]-3*kernel_size[-2])
        sampled_x = x[:,:,start_index_i:start_index_i+3*kernel_size[-2],start_index_j:start_index_j+3*kernel_size[-1] ]
        x_unfold = torch.nn.functional.unfold(sampled_x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    else:
        x_unfold = torch.nn.functional.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    mat = x_unfold.permute(0,2,1).contiguous().view(-1,x_unfold.shape[1])
    r=np.arange(mat.shape[0])
    np.random.shuffle(r)
    b = r[:max_samples]
    mat = mat[b]
    return mat

def forward_cache_activations(x, layer, key, max_samples=10000):
    act=OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        act[key]=reshape_conv_input_activation(x.clone().detach(), layer,max_samples =  max_samples)
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        act[key]=x.clone().detach()
        x = layer(x)
    else:
        x = layer(x)
    return x, act 



def forward_cache_projections(x, layer, key, alpha, max_samples=10000):
    Proj=OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        activation = reshape_conv_input_activation(x.clone().detach(), layer, max_samples =  max_samples).transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        Proj[key] =  torch.mm( mr, Ur.transpose(0,1) )
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        activation = x.clone().detach().transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        Proj[key] = torch.mm( mr, Ur.transpose(0,1) )
        x = layer(x)
    else:
        x = layer(x)
    
    
    return x, Proj 

def forward_cache_svd(x, layer, key,  max_samples=10000):
    U = OrderedDict()  
    S = OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        activation = reshape_conv_input_activation(x.clone().detach(), layer, max_samples =  max_samples).transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        U[key] = Ur
        S[key] = Sr
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        activation = x.clone().detach().transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        U[key] = Ur
        S[key] = Sr
        x = layer(x)
    else:
        x = layer(x)
    
    
    return x, U, S 
