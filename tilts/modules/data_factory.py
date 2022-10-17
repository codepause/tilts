import pandas as pd
from typing import Dict, List, Union

from . import data_cell
from . import data_graph
from tilts.apis.abstract_api import DataApi


class DataFactory:
    """ """
    data_api: DataApi
    cached_data: Dict[str, pd.DataFrame]
    pair_name_delimiter: str
    available_api_pairs: List[str]

    pairs_graph: data_graph.PairsGraph
    reachable_pairs_paths: Dict[str, List[str]]

    available_pairs: List[str]

    def __init__(self, data_api: DataApi, pair_name_delimiter: str = None):
        """

        Args:
            param data_api(DataApi): api from where we take data

        """
        self.data_api = data_api
        self.cached_data = dict()  # dict of pd.DataFrames
        if pair_name_delimiter is None:
            self.pair_name_delimiter = self.data_api.get_pair_name_delimiter()
        else:
            self.pair_name_delimiter = pair_name_delimiter
        self.available_api_pairs = self.data_api.get_available_pairs()

        self.pairs_graph = data_graph.PairsGraph(self.available_api_pairs)
        self.reachable_pairs_paths = self.pairs_graph.compute_reachable_pairs_paths(
            pair_name_delimiter=self.pair_name_delimiter)

        self.available_pairs: list = self.get_available_pairs()

    def _get_reachable_pairs_paths(self) -> Dict[str, List[str]]:
        """

        Returns:
            Dict[str, List[str]]: Dict of pairs and how to reach them

        """
        return self.reachable_pairs_paths

    def get_available_pairs(self) -> List[str]:
        """ """
        return list(self.reachable_pairs_paths.keys())

    def get_available_api_pairs(self) -> List[str]:
        """ """
        return self.data_api.get_available_pairs()

    def __update_cached_data(self, requested_api_pair: str, **kwargs):
        """
        Call to api and update cached data

        Args:
            requested_api_pair(str): Pair name to update
            kwargs(dict): Additional params

        """
        cached_pair_data = self.cached_data.get(requested_api_pair, pd.DataFrame())
        api_pair_data = self.data_api.get_pair_data(requested_api_pair, **kwargs)
        assert api_pair_data.columns.nlevels == 3  # check if data looks like workable
        updated_pair_data = pd.concat([cached_pair_data, api_pair_data], axis=0)
        # remove duplicated data
        updated_pair_data_deduplicated_index = ~updated_pair_data.index.duplicated(keep='first')
        updated_pair_data = updated_pair_data.loc[updated_pair_data_deduplicated_index]
        updated_pair_data.sort_index(inplace=True)
        self.cached_data[requested_api_pair] = updated_pair_data

    def get_pair_data(self, pair_name: Union[str, tuple], update_cache: bool = True, force_clear_cache: bool = False,
                      **kwargs) -> pd.DataFrame:
        """
        If cache pair is empty while data requested from api, cache update is forced

        Args:
            pair_name(str): Pair name to request
            update_cache(bool): If to update cached data with new call from api. \
                False useful in test/2nd call (Default value = True)
            force_clear_cache(bool): If to clear cache after request
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Requested pair_name data

        """
        if isinstance(pair_name, str):
            pair_name = (pair_name, )

        if pair_name[0] not in self.available_pairs:
            return pd.DataFrame()
            # raise Exception(f'{pair_name} not available in api')

        # get pairs we need to get from api to complete request
        if pair_name[0] not in self.cached_data:
            update_cache = True

        if update_cache:
            requested_api_pairs = self.reachable_pairs_paths[pair_name[0]]
            for requested_api_pair in requested_api_pairs:
                self.__update_cached_data(requested_api_pair, **kwargs)

        pair_cell = data_cell.DataCell(self, pair_name[0])

        # if to clear cache after request. Thinking about how is it useful
        if force_clear_cache:
            self.clear_cache()

        return pair_cell.as_pandas(pair_name)

    def get_cached_pair_data(self, pair_name: str, **kwargs) -> pd.DataFrame:
        """

        Args:
            pair_name(str): Pair name to request
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Cached data for pair_name

        """
        return self.get_pair_data(pair_name, update_cache=False, **kwargs)

    def clear_cache(self):
        """Clear cached data"""
        self.cached_data = dict()


if __name__ == '__main__':
    pass
