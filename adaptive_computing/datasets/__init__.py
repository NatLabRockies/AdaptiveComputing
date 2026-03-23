"""
Datasets handle data.
"""

from adaptive_computing.datasets.variables import ContinuousVariable, OrderedVariable, CategoricalVariable
from adaptive_computing.datasets.base import DatasetBase

# Optional Hero import - only available if hero package is installed
try:
    from adaptive_computing.datasets.hero import HeroDataset
except ImportError:
    # Hero not available - this is expected in basic AC environment
    pass