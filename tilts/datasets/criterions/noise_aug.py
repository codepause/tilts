import numpy as np

from tilts.datasets.meta import FlatCriterion
from tilts.datasets.meta import Block


class NoiseAug(FlatCriterion):
    def __init__(self):
        # ACTUALLY FOR TEST PURPOSES
        # BETTER CHANGE TO RANDOM IN data_transforms.
        pass

    def apply(self, block: 'Block', **kwargs) -> 'Dataframe':
        data = block.data + np.random.random(block.data.shape)
        return block.spawn(data)
