from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.sigdet import SigDetDataset
from .sample.selfdet import SelfDetDataset
from .dataset.ROD import ROD

dataset_factory = {
  'ROD': ROD,
}

_sample_factory = {
  'sigdet': SigDetDataset,
  'selfdet': SelfDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset