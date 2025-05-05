from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.selfdet import SelfDetDataset
from .sample.eiss import EISSDataset
from .sample.TC import TCDataset
from .sample.simclr import SimCLRDataset
from .dataset.ROD import ROD

dataset_factory = {
  'ROD': ROD,
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'selfdet': SelfDetDataset,
  'eiss': EISSDataset,
  'TC': TCDataset,
  'simclr': SimCLRDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset