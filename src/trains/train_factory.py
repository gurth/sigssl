from .ctdet import CtdetTrainer
from .eiss import EnergyInspiredSelfSupervisedPretrain
from .TC import TCPretrain
from .simclr import SimCLRPretrain

train_factory = {
  'ctdet': CtdetTrainer,
  'selfdet': CtdetTrainer,
  'eiss': EnergyInspiredSelfSupervisedPretrain,
  "TC": TCPretrain,
  "simclr": SimCLRPretrain,
}

