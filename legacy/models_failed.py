# Don't consider this file. Here I tried to implement ResNet in Syft, but it seems it will not work since BasicBlock has to inherit nn.Module for it to work with torch, but then it will not work in syft because it needs to inherit sy.Module for that and I don't want to investigate a fix for this! Probably takes to long.
# Inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import syft as sy
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['ResNet', 'resnet18']

class BasicBlock(sy.Module):
    expansion =  1
    
    def __init__(
        self,
        torch_ref,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[sy.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., sy.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__(torch_ref=torch_ref)
        if norm_layer is None:
            norm_layer = self.torch_ref.nn.BatchNorm2d
        
        self.conv1 = self.torch_ref.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                              padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = self.torch_ref.nn.ReLU(inplace=True)
        self.conv2 = self.torch_ref.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                              padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(sy.Module):
    
    def __init__(
        self,
        torch_ref,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 6,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., sy.Module]] = None,
        ):
        super(ResNet, self).__init__(torch_ref=torch_ref)
        if norm_layer is None:
            norm_layer = self.torch_ref.nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = self.torch_ref.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                              padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = self.torch_ref.nn.ReLU(inplace=True)
        self.maxpool = self.torch_ref.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = self.torch_ref.nn.AdaptiveAvgPool((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, self.troch_ref.nn.Conv2d):
                self.torch_ref.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (self.troch_ref.nn.BatchNorm2d, self.troch_ref.nn.GroupNorm)):
                self.troch_ref.nn.init.constant_(m.weight, 1)
                self.troch_ref.nn.init.constant_(m.biasm, 0)
                
    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                   stride: int = 1):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation

        downsample = None

        if self.inplanes != planes * block.expansion:
            downsample = self.troch_ref.nn.Sequential(
                self.troch_ref.nn.Conv2d(self.inplanes, planes * block.expansion,
                                         kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.torch_ref, self.inplanes, planes, stride, downsample, self.groups, 
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.torch_ref, self.inplanes, planes, groups=self.groups,
                               base_width=self.base_width, dilation=self.dilation,
                               norm_layer=norm_layer))

        return self.torch_ref.nn.Sequential(*layers)
        
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.torch_ref.flatten(x, 1)
        x = self.fc(x)

        return x

    def froward(self, x):
        return self._forward_impl(x)

        
def _resnet(
    torch_ref,
    arch: str,
    block: Type[BasicBlock],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(torch_ref, block, layers, **kwargs)
    return model
    
def resnet18(torch_ref, **kwargs: Any) -> ResNet:
    return _resnet(torch_ref, 'resnet18', BasicBlock, [2, 2, 2, 2])