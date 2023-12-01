
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from .model_utils import reshape_conv_input_activation

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

    def get_activations(self, x):
        identity = x
        act={"pre":OrderedDict(), "post":OrderedDict()}
        act["pre"]["conv1"] =reshape_conv_input_activation(deepcopy(x.clone().detach()), self.conv1).cpu().numpy()
        out = self.conv1(x)
        act["post"]["conv1"] =deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = F.relu(self.bn1(out))
        act["pre"]["conv2"] =reshape_conv_input_activation(deepcopy(out.clone().detach()), self.conv2).cpu().numpy()
        out = self.conv2(out)
        act["post"]["conv2"] =deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = self.bn2(out) 
        for layer in self.downsample: 
            if isinstance(layer, nn.Conv2d):
                act["pre"]["downsample"] =reshape_conv_input_activation(deepcopy(identity.clone().detach()), layer).cpu().numpy()
                identity =  layer(identity)
                act["post"]["downsample"] =deepcopy(identity.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, identity.shape[1]))
            else:
                identity =  layer(identity)
        out +=identity
        out = F.relu(out)
        return act, out
    
    def project_weights(self, projection_mat_dict):
        self.conv1.weight.data = torch.mm(projection_mat_dict["post"]["conv1"].transpose(0,1) ,torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["pre"]["conv1"].transpose(0,1))).view_as(self.conv1.weight.data)
        if self.conv1.bias is not None:
            self.conv1.bias.data = torch.mm( self.conv1.bias.data.unsqueeze(0), projection_mat_dict["post"]["conv1"]).squeeze(0)
        self.conv2.weight.data = torch.mm(projection_mat_dict["post"]["conv2"].transpose(0,1) ,torch.mm(self.conv2.weight.data.flatten(1), projection_mat_dict["pre"]["conv2"].transpose(0,1))).view_as(self.conv2.weight.data)
        if self.conv2.bias is not None:
            self.conv2.bias.data = torch.mm( self.conv2.bias.data.unsqueeze(0), projection_mat_dict["post"]["conv2"]).squeeze(0)
        for layer in self.downsample:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = torch.mm(projection_mat_dict["post"]["downsample"].transpose(0,1) ,torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"]["downsample"].transpose(0,1))).view_as(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data = torch.mm( layer.bias.data.unsqueeze(0), projection_mat_dict["post"]["downsample"]).squeeze(0)
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

    def get_activations(self, x):
        identity = x
        act={"pre":OrderedDict(), "post":OrderedDict()}
        act["pre"]["conv1"] =reshape_conv_input_activation(deepcopy(x.clone().detach()), self.conv1).cpu().numpy()
        out = self.conv1(x)
        act["post"]["conv1"] =deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = F.relu(self.bn1(out))
        act["pre"]["conv2"] =reshape_conv_input_activation(deepcopy(out.clone().detach()), self.conv2).cpu().numpy()
        out = self.conv2(out)
        act["post"]["conv2"] =deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = F.relu(self.bn2(out) )
        act["pre"]["conv3"] =reshape_conv_input_activation(deepcopy(out.clone().detach()), self.conv3).cpu().numpy()
        out = self.conv3(out)
        act["post"]["conv3"] =deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = self.bn3(out)
        for layer in self.downsample: 
            if isinstance(layer, nn.Conv2d):
                act["pre"]["downsample"] =reshape_conv_input_activation(deepcopy(identity.clone().detach()), layer).cpu().numpy()
                identity =  layer(identity)
                act["post"]["downsample"] =deepcopy(identity.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, identity.shape[1]))
            else:
                identity =  layer(identity)
        out +=identity
        out = F.relu(out)
        return act, out

    def project_weights(self, projection_mat_dict):
        self.conv1.weight.data = torch.mm(projection_mat_dict["post"]["conv1"].transpose(0,1) ,torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["pre"]["conv1"].transpose(0,1))).view_as(self.conv1.weight.data)
        if self.conv1.bias is not None:
            self.conv1.bias.data = torch.mm( self.conv1.bias.data.unsqueeze(0), projection_mat_dict["post"]["conv1"]).squeeze(0)
        self.conv2.weight.data = torch.mm(projection_mat_dict["post"]["conv2"].transpose(0,1) ,torch.mm(self.conv2.weight.data.flatten(1), projection_mat_dict["pre"]["conv2"].transpose(0,1))).view_as(self.conv2.weight.data)
        if self.conv2.bias is not None:
            self.conv2.bias.data = torch.mm( self.conv2.bias.data.unsqueeze(0), projection_mat_dict["post"]["conv2"]).squeeze(0)
        self.conv3.weight.data = torch.mm(projection_mat_dict["post"]["conv3"].transpose(0,1) ,torch.mm(self.conv3.weight.data.flatten(1), projection_mat_dict["pre"]["conv3"].transpose(0,1))).view_as(self.conv3.weight.data)
        if self.conv3.bias is not None:
            self.conv3.bias.data = torch.mm( self.conv3.bias.data.unsqueeze(0), projection_mat_dict["post"]["conv3"]).squeeze(0)
        if self.downsample:
            for layer in self.downsample:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = torch.mm(projection_mat_dict["post"]["downsample"].transpose(0,1) ,torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"]["downsample"].transpose(0,1))).view_as(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data = torch.mm( layer.bias.data.unsqueeze(0), projection_mat_dict["post"]["downsample"]).squeeze(0)
                    break
                else:
                    continue
        
        
class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
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
        out =  F.log_softmax(out, dim=1)
        return out
    
    def get_activations(self, x):      
        act={"pre":OrderedDict(), "post":OrderedDict()}
        block_ind=0
        fc_ind = 0
        act["pre"][f"conv1"]=reshape_conv_input_activation(deepcopy(x.clone().detach()), self.conv1).cpu().numpy()
        out = self.conv1(x)
        act["post"][f"conv1"]=deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = F.relu(self.bn1(out))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_act, out = block.get_activations(out)
                for loc in block_act.keys():
                    for name in block_act[loc].keys():
                        act[loc][f"block{block_ind}.{name}"] = block_act[loc][name]
                block_ind+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        act["pre"][f"fc{fc_ind}"]=deepcopy(out.clone().detach().cpu().numpy())
        out = self.fc(out)
        act["post"][f"fc{fc_ind}"]=deepcopy(out.clone().detach().cpu().numpy())        
        return act
    
    def project_weights(self, projection_mat_dict):
        block_ind=0
        fc_ind = 0
        self.conv1.weight.data = torch.mm(projection_mat_dict["post"]["conv1"].transpose(0,1) ,torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["pre"]["conv1"].transpose(0,1))).view_as(self.conv1.weight.data)
        if self.conv1.bias is not None:
            self.conv1.bias.data  = torch.mm(self.conv1.bias.data.unsqueeze(0),projection_mat_dict["post"]["conv1"]).squeeze(0)
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_name = f"block{block_ind}"
                block_proj_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
                for loc in projection_mat_dict.keys():
                    for act in projection_mat_dict[loc].keys():
                        if block_name == act.split(".")[0]:
                            block_proj_mat_dict[loc][".".join(act.split(".")[1:])] = projection_mat_dict[loc][act]
                block.project_weights(block_proj_mat_dict)
                block_ind+=1                
        self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict["pre"][f"fc{fc_ind}"].transpose(0,1) )
                    
        # self.fc.weight.data = torch.mm(projection_mat_dict["post"][f"fc{fc_ind}"].transpose(0,1) ,torch.mm(self.fc.weight.data, projection_mat_dict["pre"][f"fc{fc_ind}"].transpose(0,1) ))
        # if self.fc.bias is not None:
        #     self.fc.bias.data = torch.mm(self.fc.bias.data.unsqueeze(0),projection_mat_dict["post"][f"fc{fc_ind}"]).squeeze(0)
        return 
    

    

class ResNet_imagenet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
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
        out =  F.log_softmax(out, dim=1)
        return out
    
    def get_activations(self, x):      
        act={"pre":OrderedDict(), "post":OrderedDict()}
        block_ind=0
        fc_ind = 0
        act["pre"][f"conv1"]=reshape_conv_input_activation(deepcopy(x.clone().detach()), self.conv1).cpu().numpy()
        out = self.conv1(x)
        act["post"][f"conv1"]=deepcopy(out.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, out.shape[1]))
        out = self.maxpool(F.relu(self.bn1(out)))
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_act, out = block.get_activations(out)
                for loc in block_act.keys():
                    for name in block_act[loc].keys():
                        act[loc][f"block{block_ind}.{name}"] = block_act[loc][name]
                block_ind+=1
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        act["pre"][f"fc{fc_ind}"]=deepcopy(out.clone().detach().cpu().numpy())
        out = self.fc(out)
        act["post"][f"fc{fc_ind}"]=deepcopy(out.clone().detach().cpu().numpy())        
        return act
    
    def project_weights(self, projection_mat_dict):
        # ind=1
        block_ind=0
        fc_ind = 0
        self.conv1.weight.data = torch.mm(projection_mat_dict["post"]["conv1"].transpose(0,1),torch.mm(self.conv1.weight.data.flatten(1), projection_mat_dict["pre"]["conv1"].transpose(0,1))).view_as(self.conv1.weight.data)
        if self.conv1.bias is not None:
            self.conv1.bias.data  = torch.mm(self.conv1.bias.data.unsqueeze(0),projection_mat_dict["post"]["conv1"]).squeeze(0)
        for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in group:
                block_name = f"block{block_ind}"
                block_proj_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
                for loc in projection_mat_dict.keys():
                    for act in projection_mat_dict[loc].keys():
                        if block_name == act.split(".")[0]:
                            block_proj_mat_dict[loc][".".join(act.split(".")[1:])] = projection_mat_dict[loc][act]
                block.project_weights(block_proj_mat_dict)
                block_ind+=1                
                    
        self.fc.weight.data = torch.mm(self.fc.weight.data, projection_mat_dict["pre"][f"fc{fc_ind}"].transpose(0,1) )
                    
        # self.fc.weight.data = torch.mm(projection_mat_dict["post"][f"fc{fc_ind}"].transpose(0,1) ,torch.mm(self.fc.weight.data, projection_mat_dict["pre"][f"fc{fc_ind}"].transpose(0,1) ))
        # if self.fc.bias is not None:
        #     self.fc.bias.data = torch.mm(self.fc.bias.data.unsqueeze(0),projection_mat_dict["post"][f"fc{fc_ind}"]).squeeze(0)
        return 
def ResNet18(num_classes=10, dataset = "imagenet"):
    if dataset.lower() == "imagenet" or dataset.lower() == "imagenette":
        ResNet = ResNet_imagenet
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10, dataset = "imagenet"):
    if dataset.lower() == "imagenet" or dataset.lower() == "imagenette":
        ResNet = ResNet_imagenet
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10, dataset = "imagenet"):
    if dataset.lower() == "imagenet" or dataset.lower() == "imagenette":
        ResNet = ResNet_imagenet
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10, dataset = "imagenet"):
    if dataset.lower() == "imagenet" or dataset.lower() == "imagenette":
        ResNet = ResNet_imagenet
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10, dataset = "imagenet"):
    if dataset.lower() == "imagenet" or dataset.lower() == "imagenette":
        ResNet = ResNet_imagenet
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()