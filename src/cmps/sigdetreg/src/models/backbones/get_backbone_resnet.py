import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from ..utils.nested_tensor import NestedTensor

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # if name in return_layers:
            #     del return_layers[name]
            #     if not return_layers:
            #         break
        super().__init__(layers)
        self.return_layers = return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [2, 4, 8]
            self.num_channels = [64, 128, 256]
            # self.num_channels = [256, 512, 1024]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [8]
            self.num_channels = [256]
            # self.num_channels = [1024]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        if isinstance(tensor_list, NestedTensor):
            xs = self.body(tensor_list.tensors.permute(0, 2, 1))
            out: Dict[str, NestedTensor] = {}
            for name, x in xs.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-1:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
        else:
            out = self.forward_non_nested(tensor_list)
        return out

    def forward_non_nested(self, tensors):
        xs = self.body(tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out

class Backbone(BackboneBase):
    def __init__(self, backbone: nn.Module,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 load_backbone: str = ''
                ):
        if load_backbone != '':
            backbone.load_state_dict(torch.load(load_backbone))

        super().__init__(backbone, train_backbone, return_interm_layers)



class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_channels = 32

        self.num_channels = [256]

        self.conv1 = nn.Conv1d(2, 32, kernel_size=31, stride=16, padding=15, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool1d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = torch.flatten(out, 1)

        # out = out.permute(0, 2, 1)
        return out


def ResNet18_1D():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])


def get_resnet18(arg):
    backbone = ResNet18_1D()
    return backbone

def get_resnet50(arg):
    backbone = resnet50()
    return backbone


from ..networks.detr_utils.position_encoding import build_position_encoding
from .utils import Joiner

def get_backbone_resnet18(arg):
    position_embedding = build_position_encoding(arg)

    train_backbone = arg.lr_backbone > 0
    return_interm_layers = arg.masks or (arg.num_feature_levels > 1)

    backbone = get_resnet18(arg)
    backbone = Backbone(backbone, train_backbone, return_interm_layers,
                        load_backbone=arg.load_backbone)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model


from .swav_resnet50 import resnet50, resnet50w2, resnet50w4, resnet50w5
def get_backbone_resnet50(arg):
    position_embedding = build_position_encoding(arg)

    train_backbone = arg.lr_backbone > 0
    return_interm_layers = arg.masks or (arg.num_feature_levels > 1)

    backbone = get_resnet50()
    backbone = Backbone(backbone, train_backbone, return_interm_layers,
                        load_backbone=arg.load_backbone)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model