from .meanfield_optimizer import MFVI
from .meanfield_optimizer import RandomSAM
from .meanfield_optimizer import MixSAM
from .meanfield_optimizer import VSAM

__version_info_ = (0, 1, 0)
__version__ = '.'join(map(str, __version_info__))