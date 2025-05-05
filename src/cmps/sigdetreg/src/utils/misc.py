import torch
from torch import Tensor
from typing import Optional, List

from models.utils.nested_tensor import nested_tensor_from_tensor_list

import numpy as np

# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask
#
#     def to(self, device, non_blocking=False):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device, non_blocking=non_blocking)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)
#
#     def record_stream(self, *args, **kwargs):
#         self.tensors.record_stream(*args, **kwargs)
#         if self.mask is not None:
#             self.mask.record_stream(*args, **kwargs)
#
#     def decompose(self):
#         return self.tensors, self.mask
#
#     def __repr__(self):
#         return str(self.tensors)
#
# def _max_by_axis(the_list):
#     # type: (List[List[int]]) -> List[int]
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes
#
# def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
#     # TODO: 1D
#     if tensor_list[0].ndim == 2:
#         if isinstance(tensor_list[0], np.ndarray):
#             tensor_list = [torch.tensor(t) for t in tensor_list]
#
#         max_size = _max_by_axis([list(tensor.shape) for tensor in tensor_list])
#         batch_shape = [len(tensor_list)] + max_size
#         b, l, c = batch_shape
#         dtype = tensor_list[0].dtype
#         device = tensor_list[0].device
#         tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
#         mask = torch.ones((b, l), dtype=torch.bool, device=device)
#         for i, (tensor_, pad_tensor, m) in enumerate(zip(tensor_list, tensor, mask)):
#             pad_tensor[: tensor_.shape[0], : tensor_.shape[1]].copy_(tensor_)
#             m[: tensor_.shape[0]] = False
#     else:
#         raise ValueError('not supported')
#     return NestedTensor(tensor, mask)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def get_world_size():
    return 1

def is_dist_avail_and_initialized():
    return False

def is_main_process():
    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


