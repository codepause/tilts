import pandas as pd
from typing import Dict, List

from tilts.apis.abstract_api import DataApi
from tilts.modules.data_factory import DataFactory
from tilts.transforms import transforms as trs


class DataLoader(DataFactory):
    """ """

    def __init__(self, data_api: DataApi, pair_name_delimiter: str = None):
        super(DataLoader, self).__init__(data_api, pair_name_delimiter)

    def get_single_pair_data_with_transforms(self, pair_name: str,
                                             transforms: tuple = (), update_cache=True, **kwargs) -> pd.DataFrame:
        """

        Args:
            pair_name(str): Requested pair name
            transforms(tuple): Tuple of transforms to apply
            update_cache(bool, optional): If to update cached data with request (Default value = True)
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Pair_name data frame with applied transforms

        """
        if not transforms:
            transforms = [trs.Eye()]
        pair_data = self.get_pair_data(pair_name, update_cache=update_cache, **kwargs)
        concat_transform = trs.Concat(transforms)
        return concat_transform(pair_data)

    def get_pairs_data_with_transforms(self, pair_names: List[str] = None, transforms: Dict[str, tuple] = None,
                                       update_cache=True, **kwargs) -> pd.DataFrame:
        """

        Args:
            pair_names(list): Requested pair name
            transforms(dict): Tuple of transforms to apply mapped to names
            update_cache(bool, optional): If to update cached data with request (Default value = True)
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Concatenated data frames

        """
        transforms = transforms if transforms is not None else dict()
        pair_names = pair_names if pair_names is not None else list()
        if not pair_names:
            pair_names = self.data_api.get_available_pairs()
            # return pd.DataFrame()

        dfs = list()
        for pair_name in pair_names:
            pair_data = self.get_single_pair_data_with_transforms(pair_name, transforms.get(pair_name, []),
                                                                  update_cache=update_cache,
                                                                  **kwargs)
            dfs.append(pair_data)
        return_df = pd.concat(dfs, axis=1)
        return return_df

    def get_cached_pair_data_with_transforms(self, pair_name: str,
                                             transforms: tuple = (), **kwargs) -> pd.DataFrame:
        """

        Same as standard call but cached data.

        Args:
            pair_name(str): Requested pair name
            transforms(tuple): Tuple of transforms to apply
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Pair_name data frame with applied transforms

        """
        return self.get_single_pair_data_with_transforms(pair_name, transforms, update_cache=False, **kwargs)

    def get_cached_pairs_data_with_transforms(self, pair_names: List[str] = None,
                                              transforms: Dict[str, tuple] = None, **kwargs) -> pd.DataFrame:
        """

        Same as standard call but cached data.

        Args:
            pair_names(list): Requested pair name
            transforms(tuple): Tuple of transforms to apply
            kwargs(dict): Additional params

        Returns:
            pd.DataFrame: Concatenated data frames

        """
        return self.get_pairs_data_with_transforms(pair_names, transforms, update_cache=False, **kwargs)


if __name__ == '__main__':
    pass
