from .noiser import Noiser
from .radom_mask import RandomMask
from .flip_iq import FlipIQ
from .freq_offset import FrequencyOffset

augmenter_factory ={
    "noise": Noiser,
    "flip_iq": FlipIQ,
    "freq_offset": FrequencyOffset,
    "random_mask": RandomMask,
}

