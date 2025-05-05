import torch
from torch import Tensor
from typing import Optional, List

import numpy as np

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO: 1D
    if tensor_list[0].ndim == 2:
        if isinstance(tensor_list[0], np.ndarray):
            tensor_list = [torch.tensor(t) for t in tensor_list]

        max_size = _max_by_axis([list(tensor.shape) for tensor in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, l, c = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, l), dtype=torch.bool, device=device)
        for i, (tensor_, pad_tensor, m) in enumerate(zip(tensor_list, tensor, mask)):
            pad_tensor[: tensor_.shape[0], : tensor_.shape[1]].copy_(tensor_)
            m[: tensor_.shape[0]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)