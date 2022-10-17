import pandas as pd

from tilts.datasets.meta import FlatCriterion


class PdToNpCriterion(FlatCriterion):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, block: 'Block', **kwargs) -> 'Block':
        if isinstance(block.data, pd.DataFrame):
            return block.spawn(block.data.values, index=block.index, columns=block.columns)
        else:
            return block
