
import torch

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    # TODO: implement the evaluation.
    pass

@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # TODO: implement the visualization.
    pass