import os
import pandas as pd
from typing import List

from .abstract_api import DataApi


class IBCsvDataApiWeekly(DataApi):
    """ """
    csv_folder_path: str
    pair_name_delimiter: str
    available_pairs: List[str]

    def __init__(self, csv_folder_path: str, pair_name_delimiter='_'):
        self.csv_folder_path = csv_folder_path  # path to folder with ASK BID
        self.pair_name_delimiter = pair_name_delimiter

        self.available_pairs = self._get_available_pairs(self.csv_folder_path)

    def get_available_pairs(self) -> List[str]:
        """ """
        return self.available_pairs

    @staticmethod
    def _get_available_pairs(csv_folder_path: str) -> List[str]:
        """

        Args:
            csv_folder_path(str): path to csv file

        Returns:
            List[str]: Available pairs from file

        """
        assert os.path.exists(csv_folder_path)
        available_pairs = list()
        for spread in ['ASK', 'BID']:
            filenames = os.listdir(os.path.join(csv_folder_path, spread))
            filenames = [name.split('.')[0] for name in filenames]
            available_pairs.append(set(filenames))
        return sorted(list(set.intersection(*available_pairs)))

    @staticmethod
    def _load_csv(csv_file_path: str):
        if not os.path.exists(csv_file_path):
            return pd.DataFrame()
        names = ['human_timestamp', 'Open', 'High', 'Low', 'Close']
        charts = ['Open', 'High', 'Low', 'Close']
        raw_data = pd.read_csv(csv_file_path, header=1, names=names, index_col=False)
        idx = pd.to_datetime(raw_data['human_timestamp'])
        values = [raw_data[chart] for chart in charts]
        raw_data = pd.concat(values, keys=charts, axis=1)
        raw_data.index = idx.values
        return raw_data

    def _filter_names_to_load(self, names: list, load_from_date: pd.Timestamp):
        new_names = list()
        for name in names:
            if pd.to_datetime(name, format="%Y%m%d_%H%M%S") > load_from_date:
                new_names.append(name)

        return new_names

    def _is_date_valid(self, week_name: str, load_from_date: pd.Timestamp):
        name = week_name.split('.')[0]
        if pd.to_datetime(name, format="%Y%m%d_%H%M%S") > load_from_date:
            return True
        else:
            return False

    def _load_csv_pair_data(self, csv_folder_path: str, pair_name: str, **kwargs) -> pd.DataFrame:
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

        load_from_date = kwargs.get('load_from_date', None)
        if load_from_date is None:
            load_from_date = pd.Timestamp(0)

        for spread in spreads:
            spread_path = os.path.join(csv_folder_path, spread.upper(), pair_name)
            csv_file_paths = [os.path.join(spread_path, week_name) for week_name in os.listdir(spread_path) if
                              self._is_date_valid(week_name, load_from_date)]
            spread_data = list()
            for csv_file_path in csv_file_paths:
                spread_data.append(self._load_csv(csv_file_path))
            raw_data = pd.concat(spread_data)
            raw_data = raw_data.sort_index()
            if kwargs.get('check_for_duplicates', True):
                raw_data = raw_data.loc[~raw_data.index.duplicated(keep='first')]
            raw_data = pd.concat([raw_data], keys=[spread], axis=1)
            data.append(raw_data)

        data = pd.concat(data, axis=1).dropna()
        data = pd.concat([data], axis=1, keys=[pair_name])
        return data

    def get_pair_data(self, pair_name: str, **kwargs) -> pd.DataFrame:
        """ """
        return self._load_csv_pair_data(self.csv_folder_path, pair_name, **kwargs)

    def get_pair_name_delimiter(self) -> str:
        """ """
        return self.pair_name_delimiter


if __name__ == '__main__':
    pass
