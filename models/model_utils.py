import numpy as np
import torch 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/float(stride[0])+1)), int(np.floor((Lin[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/float(stride[1])+1))

# def reshape_conv_input_activation(x, conv_layer=None, in_channels=None, out_channels=None, kernel_size=3, stride=1, padding=0, dilation=1):
#     ### LEGACY CODE - Uses for loop for computation!
#     if conv_layer:
#         in_channels = conv_layer.in_channels
#         out_channels = conv_layer.out_channels
#         kernel_size = conv_layer.kernel_size
#         stride = conv_layer.stride
#         padding =  conv_layer.padding 
#         dilation = conv_layer.dilation
#     elif in_channels is None or out_channels is None:
#         return ValueError
    
#     sh, sw = compute_conv_output_size((x.shape[-2],x.shape[-1]),kernel_size, stride, padding, dilation)
#     mat = np.zeros(( x.shape[0]*sh*sw, kernel_size[0]*kernel_size[1]*in_channels))
#     x_padded = np.pad(x, ((0,0),(0,0),(padding[0], padding[0]), (padding[1], padding[1])), "constant", constant_values= ((0,0), (0,0), (0,0), (0,0)))

#     k=0
#     for kk in range(x.shape[0]):
#         for ii in range(0, sh, stride[0]):
#             for jj in range(0, sw, stride[1]):
#                 mat[k, :] = x_padded[kk, :, ii:kernel_size[0]+ii, jj:kernel_size[1]+jj].reshape(-1)
#                 k+=1
    
#     return mat

def reshape_conv_input_activation(x, conv_layer=None, kernel_size=3, stride=1, padding=0, dilation=1):
    ### FAST CODE (Avoid for loops)
    if conv_layer:
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding =  conv_layer.padding 
        dilation = conv_layer.dilation
    x_unfold = torch.nn.functional.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    mat = x_unfold.permute(0,2,1).contiguous().view(-1,x_unfold.shape[1])
    return mat