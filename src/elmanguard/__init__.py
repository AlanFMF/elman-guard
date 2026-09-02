"""ElmanGuard -- sequence-based intrusion detection with a classic Elman RNN."""

from .config import Config, DEFAULT
from .sequences import SequenceDataset, build_sequences, split_by_day, split_random
from .synthetic import generate, order_only_dataset

__version__ = "1.0.0"
__all__ = [
    "Config", "DEFAULT", "SequenceDataset", "build_sequences",
    "split_by_day", "split_random", "generate", "order_only_dataset",
]
