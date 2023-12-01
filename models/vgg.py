'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch
from collections import OrderedDict
from copy import deepcopy
import torch.nn.functional as F
from .model_utils import reshape_conv_input_activation
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes = 10, dataset="imagenet", dropout: float = 0.5):
        super(VGG, self).__init__()
        self.features = features
        if "imagenet" in dataset:
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes),
            )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x =  F.log_softmax(x, dim=1)
        return x
    
    def get_activations(self, x):
        act={"pre":OrderedDict(), "post":OrderedDict()}
        layer_ind = 0
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                act["pre"][f"conv{layer_ind}"]=reshape_conv_input_activation(deepcopy(x.clone().detach()), layer).cpu().numpy()
            elif isinstance(layer, nn.Linear):
                act["pre"][f"fc{layer_ind}"]=deepcopy(x.clone().detach().cpu().numpy())
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                act["post"][f"conv{layer_ind}"]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))
                layer_ind+=1
            elif isinstance(layer, nn.Linear):
                act["post"][f"fc{layer_ind}"]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))
                layer_ind+=1 
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, nn.Conv2d):
                act["pre"][f"conv{layer_ind}"]=reshape_conv_input_activation(deepcopy(x.clone().detach()), layer).cpu().numpy()
            elif isinstance(layer, nn.Linear):
                act["pre"][f"fc{layer_ind}"]=deepcopy(x.clone().detach().cpu().numpy())
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                act["post"][f"conv{layer_ind}"]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))
                layer_ind+=1
            elif isinstance(layer, nn.Linear):
                act["post"][f"fc{layer_ind}"]= deepcopy(x.clone().detach().cpu().numpy()) 
                layer_ind+=1 
        self.num_layer = layer_ind
        return act
    
    def project_weights(self, projection_mat_dict):
        layer_ind = 0
        for layers in [self.features, self.classifier]:
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = torch.mm( projection_mat_dict["post"][f"conv{layer_ind}"].transpose(0,1),torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"conv{layer_ind}"].transpose(0,1)) ).view_as(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data = torch.mm(layer.bias.data.unsqueeze(0), projection_mat_dict["post"][f"conv{layer_ind}"]).squeeze(0)
                    layer_ind+=1
                elif isinstance(layer, nn.Linear):
                    # Last layer check? 
                    if layer_ind+1 == self.num_layer:
                        layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"fc{layer_ind}"].transpose(0,1)).view_as(layer.weight.data)
                        
                    else:
                        layer.weight.data = torch.mm( projection_mat_dict["post"][f"fc{layer_ind}"].transpose(0,1),torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"fc{layer_ind}"].transpose(0,1)) ).view_as(layer.weight.data)
                        if layer.bias is not None:
                            layer.bias.data = torch.mm(layer.bias.data.unsqueeze(0), projection_mat_dict["post"][f"fc{layer_ind}"]).squeeze(0)
                    layer_ind+=1
                else:
                    continue
        return 
    

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(num_classes = 10, dataset="imagenet"):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes=num_classes, dataset=dataset)


def vgg11_bn(num_classes = 10, dataset="imagenet"):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def vgg13(num_classes = 10, dataset="imagenet"):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes=num_classes, dataset=dataset)


def vgg13_bn(num_classes = 10, dataset="imagenet"):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def vgg16(num_classes = 10, dataset="imagenet"):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_classes, dataset=dataset)


def vgg16_bn(num_classes = 10, dataset="imagenet"):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def vgg19(num_classes = 10, dataset="imagenet"):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes, dataset=dataset)


def vgg19_bn(num_classes = 10, dataset="imagenet"):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes, dataset=dataset)