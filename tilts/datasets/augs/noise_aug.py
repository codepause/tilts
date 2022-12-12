import numpy as np

from tilts.datasets.meta import FlatCriterion
from tilts.datasets.meta import Block


class NoiseAug(FlatCriterion):
    def apply(self, block: 'Block', **kwargs) -> 'Dataframe':
        data = block.data + np.random.random(block.data.shape)
        return block.spawn(data)
