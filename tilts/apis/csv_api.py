import os
import csv
import numpy as np
import pandas as pd
from typing import List

from .abstract_api import DataApi


class CsvDataApi(DataApi):
    """ """
    csv_pairs_path: str
    csv_folder_path: str
    pair_name_delimiter: str
    available_pairs: List[str]

    def __init__(self, csv_pairs_path: str, csv_folder_path: str, pair_name_delimiter='_'):
        self.csv_pairs_path = csv_pairs_path
        self.csv_folder_path = csv_folder_path
        self.pair_name_delimiter = pair_name_delimiter

        self.available_pairs = self._get_available_pairs(self.csv_pairs_path)

    def get_available_pairs(self) -> List[str]:
        """ """
        return self.available_pairs

    @staticmethod
    def _get_available_pairs(csv_pairs_path: str) -> List[str]:
        """

        Args:
            csv_pairs_path(str): path to csv file

        Returns:
            List[str]: Available pairs from file

        """
        assert os.path.exists(csv_pairs_path)
        available_pairs = []
        with open(csv_pairs_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                available_pairs.extend(row)
        return available_pairs

    @staticmethod
    def _load_csv_pair_data(csv_folder_path: str, pair_name: str) -> pd.DataFrame:
        """
        Loads pd.DataFrame from csv file formatted as following:
        bid_unix_timestamp, bid_human_timestamp, bid_Open, bid_High, bid_Low, bid_Close, bid_Volume, \
        ask_unix_timestamp, ask_human_timestamp, ask_Open, ask_High, ask_Low, ask_Close, ask_Volume.

        Args:
            csv_folder_path(str): path to csv folder
            pair_name(str): pair name to load

        Returns:
            pd.DataFrame: Multi-indexed pd.DataFrame with levels: [pair_name]; [ask, bid, volume]; data_keys: list

        """
        csv_file_path = os.path.join(csv_folder_path, pair_name + '.csv')
        assert os.path.exists(csv_file_path)
        names = """bid_unix_timestamp, bid_human_timestamp, bid_Open, bid_High, bid_Low, bid_Close, bid_Volume,
        ask_unix_timestamp, ask_human_timestamp, ask_Open, ask_High, ask_Low, ask_Close, ask_Volume"""
        names = [name.strip() for name in names.split(',')]
        spreads = ['ask', 'bid']
        charts = ['Open', 'High', 'Low', 'Close']
        raw_data = pd.read_csv(csv_file_path, header=None, names=names, index_col=False)
        column_multi_index = pd.MultiIndex.from_product(
            [[pair_name], spreads, charts])
        row_index = [pd.Timestamp(timestamp) for timestamp in raw_data['bid_human_timestamp']]
        refactored_data_values = [raw_data['_'.join([spread, chart])].values for spread in spreads for chart in charts]
        refactored_data_values = np.array(refactored_data_values).T

        volume_multi_index = pd.MultiIndex.from_tuples([(pair_name, 'volume', 'Volume')])
        volume_data = pd.DataFrame(raw_data['ask_Volume'].values, columns=volume_multi_index, index=row_index)
        refactored_data = pd.DataFrame(refactored_data_values, columns=column_multi_index, index=row_index)

        return_data = pd.concat([refactored_data, volume_data], axis=1)
        return return_data

    def get_pair_data(self, pair_name: str, **kwargs) -> pd.DataFrame:
        """ """
        return self._load_csv_pair_data(self.csv_folder_path, pair_name)

    def get_pair_name_delimiter(self) -> str:
        """ """
        return self.pair_name_delimiter


if __name__ == '__main__':
    pass

