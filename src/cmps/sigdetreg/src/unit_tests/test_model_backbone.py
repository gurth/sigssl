import torch

from models.backbones.backbone_factory import backbone_factory
from models.backbones.swav_resnet50 import resnet50

def test_model_backbone_resnet18():
    model = backbone_factory['resnet']()
    assert isinstance(model, torch.nn.Module)

    inp = torch.randn(1, 2, 131072)

    out = model(inp)

    print(out.shape)
    assert out.shape == torch.Size([1, 1024, 512])


def test_model_backbone_resnet50():
    model = resnet50()
    assert isinstance(model, torch.nn.Module)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inp = torch.randn(1, 2, 131072).to(device)
    model = model.to(device)

    out = model(inp)

    print(out.shape)
    assert out.shape == torch.Size([1, 2048, 512])

def test_model_backbone_conformer():
    model = backbone_factory['conformer']()

    assert isinstance(model, torch.nn.Module)

    inp = torch.randn(1, 2, 131072)

    out = model(inp)

    print(out.shape)

def test_model_backbone():
    test_model_backbone_resnet50()
    test_model_backbone_conformer()
    test_model_backbone_resnet18()


