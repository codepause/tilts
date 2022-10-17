import os
import pandas as pd
from typing import List, Set

from .abstract_api import DataApi


class IBCsvDataApi(DataApi):
    """ """
    csv_folder_path: str
    pair_name_delimiter: str
    available_pairs: List[Set[str]]

    def __init__(self, csv_folder_path: str, pair_name_delimiter='_', tz='GMT'):
        self.csv_folder_path = csv_folder_path  # path to folder with ASK BID
        self.pair_name_delimiter = pair_name_delimiter

        self.available_pairs = self._get_available_pairs(self.csv_folder_path)
        self.tz = tz

    def get_available_pairs(self) -> List[Set[str]]:
        """ """
        return self.available_pairs

    @staticmethod
    def _get_available_pairs(csv_folder_path: str) -> List[Set[str]]:
        """

        Args:
            csv_folder_path(str): path to csv file

        Returns:
            List[str]: Available pairs from file

        """
        available_pairs = list()

        if not os.path.exists(csv_folder_path):
            return available_pairs

        for spread in ['ASK', 'BID']:
            if not os.path.exists(os.path.join(csv_folder_path, spread)):
                return available_pairs
            filenames = os.listdir(os.path.join(csv_folder_path, spread))
            filenames = [name.split('.')[0] for name in filenames]
            available_pairs.append(set(filenames))
        return sorted(list(set.intersection(*available_pairs)))

    @staticmethod
    def _load_csv_pair_data(csv_folder_path: str, pair_name: str, **kwargs) -> pd.DataFrame:
        """
        Loads pd.DataFrame from csv file formatted as following:
        human_timestamp, Open, High, Low, Close.

        Args:
            csv_folder_path(str): path to csv folder
            pair_name(str): pair name to load

        Returns:
            pd.DataFrame: Multi-indexed pd.DataFrame with levels: [pair_name]; [ask, bid, volume]; data_keys: list

        """
        spreads = ['ask', 'bid']
        data = list()
        for spread in spreads:
            csv_file_path = os.path.join(csv_folder_path, spread.upper(), pair_name + '.csv')
            if not os.path.exists(csv_file_path):
                return pd.DataFrame()
            names = ['human_timestamp', 'Open', 'High', 'Low', 'Close']
            charts = ['Open', 'High', 'Low', 'Close']
            raw_data = pd.read_csv(csv_file_path, header=0, names=names, index_col=False)
            idx = pd.to_datetime(raw_data['human_timestamp'], format="%Y%m%d  %H:%M:%S")
            values = [raw_data[chart] for chart in charts]
            raw_data = pd.concat(values, keys=charts, axis=1)
            raw_data.index = idx.values
            if kwargs.get('check_for_duplicates', False):
                raw_data = raw_data.loc[~raw_data.index.duplicated(keep='first')]
            raw_data = pd.concat([raw_data], keys=[spread], axis=1)
            data.append(raw_data)
        data = pd.concat(data, axis=1).dropna()
        data = pd.concat([data], axis=1, keys=[pair_name])
        if kwargs.get('tz'):
            data.index = data.index.tz_localize(kwargs.get('tz'))
        return data

    def get_pair_data(self, pair_name: str, **kwargs) -> pd.DataFrame:
        """ """
        return self._load_csv_pair_data(self.csv_folder_path, pair_name, tz=self.tz, **kwargs)

    def get_pair_name_delimiter(self) -> str:
        """ """
        return self.pair_name_delimiter


if __name__ == '__main__':
    pass
