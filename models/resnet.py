
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from .model_utils import reshape_conv_input_activation, forward_cache_activations, forward_cache_projections, forward_cache_svd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def get_activations(self, x, block_key, max_samples=10000):
        identity = x
        out, act = forward_cache_activations(x, self.conv1, f"{block_key}.conv1", max_samples)
        out = F.relu(self.bn1(out))
        out, layer_acts = forward_cache_activations(out, self.conv2, f"{block_key}.conv2", max_samples)
        act.update(layer_acts)           
        out = self.bn2(out) 
        for layer in self.downsample: 
            identity, layer_acts = forward_cache_activations(identity, layer, f"{block_key}.downsample", max_samples)
            act.update(layer_acts)   
        out +=identity
        out = F.relu(out)
        return act, out

    def get_svd_directions(self, x, block_key, max_samples=10000):
        identity = x
        out, U, S = forward_cache_svd(x, self.conv1, f"{block_key}.conv1", max_samples)
        out = F.relu(self.bn1(out))
        out, layer_u, layer_s = forward_cache_svd(out, self.conv2, f"{block_key}.conv2", max_samples)
        U.update(layer_u)
        S.update(layer_s)
        out = self.bn2(out) 
        for layer in self.downsample: 
            identity,  layer_u, layer_s  = forward_cache_svd(identity, layer, f"{block_key}.downsample", max_samples)
            U.update(layer_u)
            S.update(layer_s)
        out +=identity
        out = F.relu(out)
        return U, S, out

    def get_scaled_projections(self, x, block_key, alpha, max_samples=10000):
        identity = x
        out, proj = forward_cache_projections(x, self.conv1, f"{block_key}.conv1", alpha, max_samples)
        out = F.relu(self.bn1(out))
        out, layer_acts = forward_cache_projections(out, self.conv2, f"{block_key}.conv2", alpha, max_samples)
        proj.update(layer_acts)           
        out = self.bn2(out) 
        for layer in self.downsample: 
            identity, layer_acts = forward_cache_projections(identity, layer, f"{block_key}.downsample", alpha, max_samples)
            proj.update(layer_acts)   
        out +=identity
        out = F.relu(out)
        return proj, out
    
    def project_weights(self, projection_mat_dict, block_key):
        self.conv1.weight.data = torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict[f"{block_key}.conv1"].transpose(0,1)).view_as(self.conv1.weight.data)
        self.conv2.weight.data = torch.mm(self.conv2.weight.data.flatten(1), projection_mat_dict[f"{block_key}.conv2"].transpose(0,1)).view_as(self.conv2.weight.data)
        for layer in self.downsample:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict[f"{block_key}.downsample"].transpose(0,1)).view_as(layer.weight.data)
                break
            else:
                continue


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def get_activations(self, x, block_key, max_samples=10000):
        identity = x
        out, act = forward_cache_activations(x, self.conv1, f"{block_key}.conv1", max_samples)
        out = F.relu(self.bn1(out))
        out, layer_acts = forward_cache_activations(out, self.conv2, f"{block_key}.conv2", max_samples)
        act.update(layer_acts)        
        out = F.relu(self.bn2(out) )
        out, layer_acts = forward_cache_activations(out, self.conv3, f"{block_key}.conv2", max_samples)
        act.update(layer_acts)   
        out = self.bn3(out)
        for layer in self.downsample: 
            identity, layer_acts = forward_cache_activations(identity, layer, f"{block_key}.downsample", max_samples)
            act.update(layer_acts)   
        out +=identity
        out = F.relu(out)
        return act, out



    def get_activations(self, x, block_key, max_samples=10000):
        identity = x
        out, U, S = forward_cache_svd(x, self.conv1, f"{block_key}.conv1", max_samples)
        out = F.relu(self.bn1(out))
        out, layer_u, layer_s = forward_cache_activations(out, self.conv2, f"{block_key}.conv2", max_samples)
        U.update(layer_u)        
        S.update(layer_s)
        out = F.relu(self.bn2(out) )
        out, layer_u, layer_s = forward_cache_activations(out, self.conv3, f"{block_key}.conv2", max_samples)
        U.update(layer_u)        
        S.update(layer_s)
        out = self.bn3(out)
        for layer in self.downsample: 
            identity, layer_u, layer_s = forward_cache_activations(identity, layer, f"{block_key}.downsample", max_samples)
            U.update(layer_u)        
            S.update(layer_s)  
        out +=identity
        out = F.relu(out)
        return U, S, out
    def get_scaled_projections(self, x, block_key, alpha, max_samples=10000):
        identity = x
        out, proj = forward_cache_projections(x, self.conv1, f"{block_key}.conv1", max_samples)
        out = F.relu(self.bn1(out))
        out, layer_acts = forward_cache_projections(out, self.conv2, f"{block_key}.conv2", max_samples)
        proj.update(layer_acts)        
        out = F.relu(self.bn2(out) )
        out, layer_acts = forward_cache_projections(out, self.conv3, f"{block_key}.conv2", max_samples)
        proj.update(layer_acts)   
        out = self.bn3(out)
        for layer in self.downsample: 
            identity, layer_acts = forward_cache_projections(identity, layer, f"{block_key}.downsample", max_samples)
            proj.update(layer_acts)   
        out +=identity
        out = F.relu(out)
        return proj, out

    def project_weights(self, projection_mat_dict, block_key):
        self.conv1.weight.data = torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict[f"{block_key}.conv1"].transpose(0,1)).view_as(self.conv1.weight.data)
        self.conv2.weight.data = torch.mm(self.conv2.weight.data.flatten(1), projection_mat_dict[f"{block_key}.conv2"].transpose(0,1)).view_as(self.conv2.weight.data)
        self.conv3.weight.data = torch.mm(self.conv3.weight.data.flatten(1), projection_mat_dict[f"{block_key}.conv3"].transpose(0,1)).view_as(self.conv3.weight.data)
        if self.downsample:
            for layer in self.downsample:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict[f"{block_key}.downsample"].transpose(0,1)).view_as(layer.weight.data)
                    break
                else:
                    continue
        
        
class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, do_log_softmax=True):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.do_log_softmax = do_log_softmax

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if self.do_log_softmax:
            out =  F.log_softmax(out, dim=1)
        return out
    
    def get_activations(self, x, max_samples=10000):      
        block_ind=0
        fc_ind = 0
        out, act = forward_cache_activations(x, self.conv1, "conv1", max_samples)
        out = F.relu(self.bn1(out))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_acts, out = block.get_activations(out, f"block{block_ind}", max_samples)
                act.update(block_acts)  
                block_ind+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, block_acts = forward_cache_activations(out, self.fc, f"fc{fc_ind}", max_samples)
        act.update(block_acts)  
        return act

    
    
    def get_svd_directions(self, x, max_samples=10000):      
        block_ind=0
        fc_ind = 0
        out, U, S = forward_cache_svd(x, self.conv1, "conv1", max_samples)
        out = F.relu(self.bn1(out))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_u, block_s, out  = block.get_svd_directions(out, f"block{block_ind}", max_samples)
                U.update(block_u)  
                S.update(block_s)
                block_ind+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, layer_u, layer_s = forward_cache_svd(out, self.fc, f"fc{fc_ind}", max_samples)
        U.update(layer_u)  
        S.update(layer_s)
        return U,S
    def get_scaled_projections(self, x, alpha, max_samples=10000):      
        block_ind=0
        fc_ind = 0
        out, proj = forward_cache_projections(x, self.conv1, "conv1", alpha, max_samples)
        out = F.relu(self.bn1(out))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_acts, out = block.get_scaled_projections(out, f"block{block_ind}", alpha, max_samples)
                proj.update(block_acts)  
                block_ind+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, block_acts = forward_cache_projections(out, self.fc, f"fc{fc_ind}", alpha, max_samples)
        proj.update(block_acts)  
        return proj

    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        block_ind=0
        fc_ind = 0
        self.conv1.weight.data = torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["conv1"].transpose(0,1)).view_as(self.conv1.weight.data)
        
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_key = f"block{block_ind}"
                block.project_weights(projection_mat_dict, block_key)
                block_ind+=1       
        if not proj_classifier:
            self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict[f"fc{fc_ind}"].transpose(0,1) )
        else:                    
            self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict[f"fc{fc_ind}"].transpose(0,1) )
        return 
    

    

class ResNet_imagenet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, do_log_softmax=True):
        super(ResNet_imagenet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.do_log_softmax=do_log_softmax
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.do_log_softmax:
            out =  F.log_softmax(out, dim=1)
        return out
    def get_activations(self, x, max_samples=10000):      
        block_ind=0
        fc_ind = 0
        out, act = forward_cache_activations(x, self.conv1, "conv1", max_samples)
        out = self.maxpool(F.relu(self.bn1(out)))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_acts, out = block.get_activations(out, f"block{block_ind}", max_samples)
                act.update(block_acts)  
                block_ind+=1
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out, block_acts = forward_cache_activations(out, self.fc, f"fc{fc_ind}", max_samples)
        act.update(block_acts)  
        return act
    
    def get_svd_directions(self, x, max_samples=10000):   
        block_ind=0
        fc_ind = 0
        out, U, S = forward_cache_svd(x, self.conv1, "conv1", max_samples)
        out = self.maxpool(F.relu(self.bn1(out)))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_u, block_s, out = block.get_svd_directions(out, f"block{block_ind}", max_samples)
                U.update(block_u)  
                S.update(block_s)  
                block_ind+=1
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out, layer_u, layer_s = forward_cache_svd(out, self.fc, f"fc{fc_ind}", max_samples)
        U.update(layer_u)  
        S.update(layer_s)  
        return U, S

    def get_scaled_projections(self, x, max_samples=10000):   
        block_ind=0
        fc_ind = 0
        out, proj = forward_cache_projections(x, self.conv1, "conv1", alpha, max_samples)
        out = self.maxpool(F.relu(self.bn1(out)))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:                
                block_acts, out = block.get_scaled_projections(out, f"block{block_ind}", alpha, max_samples)
                proj.update(block_acts)  
                block_ind+=1
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out, block_acts = forward_cache_activations(out, self.fc, f"fc{fc_ind}", alpha, max_samples)
        proj.update(block_acts)  
        return proj
    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        block_ind=0
        fc_ind = 0
        self.conv1.weight.data = torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["conv1"].transpose(0,1)).view_as(self.conv1.weight.data)
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_key = f"block{block_ind}"
                block.project_weights(projection_mat_dict, block_key)
                block_ind+=1        
        if not proj_classifier:
            self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict[f"fc{fc_ind}"].transpose(0,1) )
        else:                    
            self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict[f"fc{fc_ind}"].transpose(0,1) )
            
        return 
           
    
def ResNet18(num_classes=1000, dataset = "imagenet", do_log_softmax=True):
    if "imagenet" in dataset.lower()  or  "vggface" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, do_log_softmax=do_log_softmax)


def ResNet34(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower()  or  "vggface" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower()  or  "vggface" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower()  or  "vggface" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower()  or  "vggface" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()