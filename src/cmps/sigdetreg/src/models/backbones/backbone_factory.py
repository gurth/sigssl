

from .get_backbone_resnet import get_backbone_resnet18, get_backbone_resnet50, get_resnet18, get_resnet50
from .get_backbone_conformer import get_backbone_conformer


backbone_factory = {
    'resnet18': get_backbone_resnet18,
    'resnet50': get_backbone_resnet50,
    'conformer': get_backbone_conformer
}

pure_backbone_factory = {
    'resnet18': get_resnet18,
    'resnet50': get_resnet50
}

def get_backbone(arg):
    backbone = backbone_factory[arg.backbone](arg)

    return backbone

def get_pure_backbone(arg):
    backbone = pure_backbone_factory[arg.backbone](arg)

    return backbone