import torch

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace for 1D data.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

def crop_bbox(feats, bbox, HH):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, L)
    - bbox: Bounding box coordinates of shape (N, 2) in the format
      [x0, x1] in the [0, 1] coordinate space.
    - HH: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 2
    bbox = torch.clamp(bbox, 0.01, 0.99)
    bbox = 2 * bbox - 1
    x0, x1 = bbox[:, 0], bbox[:, 1]
    X = tensor_linspace(x0, x1, steps=HH).view(N, HH).expand(N, HH)
    grid = X.unsqueeze(2)
    res = torch.nn.functional.grid_sample(feats.unsqueeze(3), grid.unsqueeze(3), padding_mode='border', align_corners=False)
    return res.squeeze(3)

def box_cxw_to_xx(x):
    x_c, w = x.unbind(-1)
    b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)

def box_xx_to_cxw(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]

    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]

    inter = (rb - lt).clamp(min=0)  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU for 1D boxes

    The boxes should be in [x0, x1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 1] >= boxes1[:, 0]).all()
    assert (boxes2[:, 1] >= boxes2[:, 0]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    area = (rb - lt).clamp(min=0)  # [N, M]

    giou = iou - (area - union) / area
    return giou

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, L] where N is the number of masks, L is the length.

    Returns a [N, 2] tensors, with the boxes in xx format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    l = masks.shape[-1]

    x = torch.arange(0, l, dtype=torch.float)
    x = x.unsqueeze(0)

    x_mask = (masks * x)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, x_max], 1)
