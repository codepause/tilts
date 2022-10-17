import os
import pandas as pd
from typing import List
import itertools

from tilts.apis.abstract_api import DataApi


class SingleFileCsvApi(DataApi):
    csv_path: str
    pair_name_delimiter: str

    def __init__(self, csv_path: str, pair_name_delimiter='', tz='UTC'):
        self.csv_path = csv_path
        self.pair_name_delimiter = pair_name_delimiter
        self.available_pairs = self._get_available_pairs(self.csv_path)
        self._pair_name = self.available_pairs[0]
        self.tz = tz

    def get_available_pairs(self) -> List[str]:
        return self.available_pairs

    @staticmethod
    def _get_available_pairs(csv_path: str) -> List[str]:
        """

        Args:
            csv_path(str): path to csv file

        Returns:
            List[str]: Available pairs from file

        """
        available_pairs = list()

        if not os.path.exists(csv_path):
            assert False
            # return available_pairs

        pair_name = os.path.basename(csv_path)
        pair_name = '.'.join(pair_name.split('.')[:-1])
        available_pairs.append(pair_name)
        return available_pairs

    @staticmethod
    def _load_csv_pair_data(csv_path: str, pair_name: str = None, **kwargs) -> pd.DataFrame:
        """
        Loads pd.DataFrame from csv file formatted as following:
        timestamp, Open, High, Low, Close, Volume

        Args:
            csv_folder_path(str): path to csv file
            pair_name(str): pair name to load

        Returns:
            pd.DataFrame: Multi-indexed pd.DataFrame with levels: [pair_name]; [ask, bid, volume]; data_keys: list

        """
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        spreads = ['mid', 'volume']
        ohlc = ['Open', 'High', 'Low', 'Close']
        vol = ['Volume']
        raw_data = pd.read_csv(csv_path, header=0, names=ohlc + vol, index_col=0)
        raw_data.index = pd.to_datetime(raw_data.index, unit='s')
        if kwargs.get('check_for_duplicates', False):
            raw_data = raw_data.loc[~raw_data.index.duplicated(keep='first')]

        new_columns = pd.MultiIndex.from_tuples(tuple(itertools.product([pair_name], ['mid'], ohlc)) +
                                                tuple(itertools.product([pair_name], ['volume'], vol)))
        raw_data.columns = new_columns
        if kwargs.get('tz'):
            raw_data.index = raw_data.index.tz_localize(kwargs.get('tz'))
        return raw_data

    def get_pair_data(self, pair_name: str = None, **kwargs) -> pd.DataFrame:
        """ """
        pair_name = pair_name or self._pair_name
        return self._load_csv_pair_data(self.csv_path, pair_name, tz=self.tz, **kwargs)

    def get_pair_name_delimiter(self) -> str:
        """ """
        return self.pair_name_delimiter


if __name__ == '__main__':
    pass

