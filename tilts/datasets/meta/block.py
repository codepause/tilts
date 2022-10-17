from typing import Union
from tilts.transforms.transforms import DfTool
from .criterion import Criterion

import pandas as pd
import numpy as np


class Block:
    def __init__(self, df: Union['Dataframe', 'Array'], idx: int, timestamp_idx=None, index=None, columns=None,
                 connected_block=None):
        """

        Args:
            df: dataframe to sample from
            idx: starting index of a block
            timestamp_idx: corresponding timeframe
            index: new dataframe index
            columns: new dataframe columns
            connected_block: block to connect with
        """
        self._initial_df = df
        self._idx = idx
        if timestamp_idx is None:
            self._timestamp_idx = self._initial_df.index[self._idx]
        else:
            self._timestamp_idx = timestamp_idx

        self._data = df
        self._index = df.index if isinstance(df, pd.DataFrame) else index
        self._columns = df.columns if isinstance(df, pd.DataFrame) else columns

        self._connected_block = connected_block

    def spawn(self, df: Union['Dataframe', 'Array', None], index=None, columns=None) -> 'Block':
        """
        :param df:  new values for block data
        :return: Returns Block with block.data = df;
        """
        new_block = Block(df, self.idx, timestamp_idx=self.timestamp_idx, index=index, columns=columns)
        return new_block

    @property
    def connected_block(self) -> int:
        """
        :return: For example, you need to spawn Y block with some specific conditions to X block. Returns X block
        """
        return self._connected_block

    @property
    def idx(self) -> int:
        """
        :return: initial block idx
        """
        return self._idx

    @property
    def data(self) -> 'Dataframe':
        """
        :return: current data stored in block
        """
        return self._data

    def as_pd(self) -> 'Dataframe':
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            return pd.DataFrame(self._data, columns=self.columns, index=self.index)

    def as_np(self) -> 'Array':
        if isinstance(self._data, pd.DataFrame):
            return self._data.values
        else:
            return np.array(self._data)

    @property
    def timestamp_idx(self) -> 'Timestamp':
        """
        :return: initial block timestamp
        """
        return self._timestamp_idx

    @property
    def empty(self) -> bool:
        """
        :return: True if data is None or empty df
        """
        if self._data is None:
            return True
        else:
            return False

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    def apply(self, criterion: Union['Criterion', 'DfTool'], *args, **kwargs) -> 'Block':
        if self.empty:
            return self.spawn(None)
        if isinstance(criterion, DfTool):  # pandas only
            return self.spawn(criterion(self.data, **kwargs))
        elif isinstance(criterion, Criterion):  # FlatCriterion
            return criterion(self, **kwargs)
        # TODO: actually LearnCriterion is no more needed?
        # elif isinstance(criterion, LearnCriterion):
        #     return criterion(self, **kwargs)
        else:
            raise NotImplementedError('Nor Transfrom or Criterion instance are passed through builder')

    def __len__(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)
