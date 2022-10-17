import abc
import pandas as pd
from typing import List


class SingleApi(abc.ABC):
    """ Class for a single data stream """

    @abc.abstractmethod
    def get_pair_name_delimiter(self) -> str:
        """

        Returns:
            str: Name delimiter for current api. f('USD_HUF') = '_'

        """
        pass

    @abc.abstractmethod
    def get_pair_data(self, pair_name: str, **kwargs) -> pd.DataFrame:
        """

        Args:
            pair_name(str): Requested pair name
            **kwargs(dict): K-word params

        Returns:
            pd.DataFrame: multi-index pd.DataFrame with levels: [pair_name]; [ask,bid]; data_keys: list

        """
        pass


class DataApi(SingleApi, abc.ABC):
    """Dummy class for DataApi methods."""

    @abc.abstractmethod
    def get_available_pairs(self) -> List[str]:
        """

        Returns:
            list: pairs that could be requested via DataApi

        """
        pass
