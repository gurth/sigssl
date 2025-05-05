from .sigdet import SigdetTrainer

train_factory = {
  'sigdet': SigdetTrainer,
  'selfdet': SigdetTrainer,
}