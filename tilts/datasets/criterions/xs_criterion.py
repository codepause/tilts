import pandas as pd
from typing import List, Tuple, Union

from tilts.datasets.meta import FlatCriterion


class XsCriterion(FlatCriterion):
    def __init__(self, keys: List[Union[Tuple[str, ...], str]], level: Union[int, list] = 0):
        """
        Args:
            keys: List[Tuple[str, ...]].
        """
        self.keys = keys
        if isinstance(level, int):
            self.level = [level] * len(self.keys)
        else:
            self.level = level

    def apply(self, block: 'Block', **kwargs) -> 'Block':
        """
        Args:
            block:

        Returns: Selects keys from columns via List[Tuple[str, ...]]

        """
        if isinstance(block.data, pd.DataFrame):
            return_l = [block.data.xs(key, axis=1, drop_level=False, level=self.level[idx]) for idx, key in
                        enumerate(self.keys)]
            cat_df = pd.concat(return_l, axis=1)
            return block.spawn(cat_df)
        else:
            return block.spawn(None)
