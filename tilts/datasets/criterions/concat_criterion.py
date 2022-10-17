import pandas as pd
from typing import List

from tilts.datasets.meta import LearnCriterion, Criterion

class ConcatCriterion(LearnCriterion):
    def __init__(self, criterions: List['Criterion'], axis: int = 0):
        """
        Args:
            criterions: criterions from which take data and concatenate
        """
        self.criterions = criterions
        self.axis = axis

    def fit(self, block: 'Block'):
        return self.apply(block, 'fit')

    def transform(self, block: 'Block'):
        return self.apply(block, 'transform')

    def apply(self, block: 'Block', mode: str) -> 'Block':
        """
        Args:
            block:
            mode:

        Returns: Selects keys for each of criterions and concatenate, axis=0
        Be careful with returning data if trying to concatenate numpy data (not pd.DataFrame).

        """
        crit_data = list()
        for criterion in self.criterions:
            temp_block = block.apply(criterion, mode=mode)
            if not temp_block.empty:
                crit_data.append(temp_block)
        if crit_data:
            temp_data = [temp_block.as_pd() for temp_block in crit_data]
            return block.spawn(pd.concat(temp_data, axis=self.axis))
        else:
            return block.spawn(None)

    def reset(self):
        for criterion in self.criterions:
            criterion.reset()
