import pandas as pd
import numpy as np
import re
from typing import List, Tuple

from .transforms import Invert


def _get_common_names(spread_level_df_1: List[str], spread_level_df_2: List[str]) -> List[str]:
    """

    Args:
        spread_level_df_1(List[str]): df_1 names on the spread level
        spread_level_df_2(List[str]): df_2 names on the spread level

    Returns:
        List[str]: Common names of data frames for the spread level

    """
    spread_level_union = [spread_level_df_1, spread_level_df_2]
    common_spreads = set.intersection(*[set(spread_level) for spread_level in spread_level_union])
    common_spreads = list(common_spreads)
    return common_spreads


def _get_pair_name_tuple(df: pd.DataFrame) -> Tuple[str, List[str], str]:
    """
    Get names list and its delimiter of level 0 of pd.DataFrame with 3 levels.
    pd.DataFrame['USD_CAD', [level_1], [level_2]] -> ['USD', 'CAD'], '_'

    Args:
        df(pd.DataFrame): Input data frame

    Returns:
      Tuple[str, List[str], str]: USD_CAD', ['USD', 'CAD'], '_'

    """
    pair_name: str = df.columns.get_level_values(0).unique()[0]
    pair_name_delimiter = get_pair_name_delimiter(pair_name)
    pair_names = pair_name.split(pair_name_delimiter)
    return pair_name, pair_names, pair_name_delimiter


def get_pair_name_delimiter(pair_name: str) -> str:
    """

    Args:
        pair_name(str): Pair name

    Returns:
        str: Pair name delimiter

    """
    rgx = r'[^a-zA-Z0-9]'  # like \\W but with "_"
    delimiter = re.search(rgx, pair_name).group(0)
    return delimiter


class Multiply:
    """Multiplication of two frames"""
    feature_names_to_drop: Tuple[str]

    def __init__(self, feature_names_to_drop: Tuple[str] = ('Volume',)):
        self.feature_names_to_drop = feature_names_to_drop

        # mapping for name replacement on level_1 and level_2
        self.spread_level_replace = {'ask': 'ask', 'bid': 'bid'}
        self.feature_level_replace = {'High': 'High', 'Low': 'Low'}

    def multiply_level_2(self, df_1_level_2: pd.DataFrame, df_2_level_2: pd.DataFrame) -> pd.DataFrame:
        """
        Deals with invert level 2 values (O H L C)

        Args:
            df_1_level_2(pd.DataFrame): Data frame on level 2 of df_1
            df_2_level_2(pd.DataFrame): Data frame on level 2 of df_2

        Returns:
            pd.DataFrame: Data frame with inverted level 2 values (inverted column names and values)

        """
        return_df = df_1_level_2 * df_2_level_2
        return_df = return_df.dropna(axis=1, how='all')

        new_columns = [self.feature_level_replace.get(column_name, column_name) for column_name in return_df.columns]
        return_df.columns = new_columns

        return return_df

    def multiply_level_1(self, df_1_level_1: pd.DataFrame, df_2_level_1: pd.DataFrame) -> pd.DataFrame:
        """Deals with invert level 1 values (ask, bid). Uses invert_level2

        Args:
            df_1_level_1(pd.DataFrame): Data frame that is on level 1 of df_1
            df_2_level_1(pd.DataFrame): Data frame that is on level 1 of df_2

        Returns:
            pd.DataFrame: Data frame with inverted level 2 and level 1

        """
        common_spreads = _get_common_names(df_1_level_1.columns.get_level_values(0).unique(),
                                           df_2_level_1.columns.get_level_values(0).unique())

        return_dfs = [self.multiply_level_2(df_1_level_1[spread], df_2_level_1[spread]) for spread in common_spreads]
        new_columns = [self.spread_level_replace.get(common_spread, common_spread) for common_spread in common_spreads]

        return_df = pd.concat(return_dfs, keys=new_columns, axis=1)

        return return_df

    def _drop_df_columns(self, df: pd.DataFrame, level: int = 2) -> pd.DataFrame:
        """

        Args:
          df(pd.DataFrame): Input data frame
          level(int): Level on which to drop (Default value = 2)

        Returns:
            pd.DataFrame: Data frame with dropped columns

        """
        df_columns = np.array(df.columns.get_level_values(2).unique())
        columns_to_drop_index = np.isin(df_columns, list(self.feature_names_to_drop))
        columns_to_drop = df_columns[columns_to_drop_index]
        dropped_df = df.drop(columns_to_drop, axis=1, level=level)
        return dropped_df

    @staticmethod
    def _check_if_applicable(pair_names_1: List[str], pair_names_2: List[str]) -> bool:
        """

        Args:
            pair_names_1(List[str]): Names of pair 1
            pair_names_2(List[str]): Names of pair 2

        Returns:
            bool: Is operation possible or not

        """
        return pair_names_1[-1] == pair_names_2[0]

    @staticmethod
    def _get_new_pair_name(pair_names_1: List[str], pair_names_2: List[str], pair_name_1_delimiter: str) -> str:
        """

        Args:
          pair_names_1(List[str]): Names of pair 1
          pair_names_2(List[str]): Names of pair 2
          pair_name_1_delimiter(str): Name delimiter

        Returns:
            str: New name after operation

        """
        return pair_names_1[0] + pair_name_1_delimiter + pair_names_2[-1]

    def __call__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
        """
        Dividing by levels in case of replacing level names. First drop self.feature_names_to_drop

        Args:
            df_1(pd.DataFrame): df_1 to be divided
            df_2(pd.DataFrame): df_2 to be divisor
        Returns:
            pd.DataFrame: Output data frame

        """
        pair_name_1, pair_names_1, pair_name_1_delimiter = _get_pair_name_tuple(df_1)
        pair_name_2, pair_names_2, _ = _get_pair_name_tuple(df_2)

        assert self._check_if_applicable(pair_names_1, pair_names_2)  # USD_EUR x CAD_EUR
        new_pair_name = self._get_new_pair_name(pair_names_1, pair_names_2, pair_name_1_delimiter)  # USD_CAD

        dropped_df_1 = self._drop_df_columns(df_1)
        dropped_df_2 = self._drop_df_columns(df_2)

        multiplied_df = self.multiply_level_1(dropped_df_1[pair_name_1], dropped_df_2[pair_name_2])

        return_df = pd.concat([multiplied_df], keys=[new_pair_name], axis=1)

        return return_df

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Divide:
    """Division of two frames"""
    feature_names_to_drop: Tuple[str]

    def __init__(self, feature_names_to_drop: Tuple[str] = ('Volume',)):
        self.feature_names_to_drop = feature_names_to_drop

        # mapping for name replacement on level_1 and level_2
        self.spread_level_replace = {'ask': 'bid', 'bid': 'ask'}
        self.feature_level_replace = {'High': 'High', 'Low': 'Low'}

    def divide_level_2(self, df_1_level_2: pd.DataFrame, df_2_level_2: pd.DataFrame) -> pd.DataFrame:
        """
        Deals with invert level 2 values (O H L C)

        Args:
            df_1_level_2(pd.DataFrame): Data frame on level 2 of df_1
            df_2_level_2(pd.DataFrame): Data frame on level 2 of df_2

        Returns:
            pd.DataFrame: Data frame with inverted level 2 values (inverted column names and values)

        """

        return_df = df_1_level_2 / df_2_level_2

        return_df = return_df.dropna(axis=1, how='all')

        new_columns = [self.feature_level_replace.get(column_name, column_name) for column_name in return_df.columns]
        return_df.columns = new_columns

        return return_df

    def divide_level_1(self, df_1_level_1: pd.DataFrame, df_2_level_1: pd.DataFrame) -> pd.DataFrame:
        """Deals with invert level 1 values (ask, bid). Uses invert_level2

        Args:
            df_1_level_1(pd.DataFrame): Data frame that is on level 1 of df_1
            df_2_level_1(pd.DataFrame): Data frame that is on level 1 of df_2

        Returns:
            pd.DataFrame: Data frame with inverted level 2 and level 1

        """
        common_spreads = _get_common_names(df_1_level_1.columns.get_level_values(0).unique(),
                                           df_2_level_1.columns.get_level_values(0).unique())

        return_dfs = [self.divide_level_2(df_1_level_1[common_spread], df_2_level_1[common_spread]) for common_spread in
                      common_spreads]
        new_columns = [self.spread_level_replace.get(common_spread, common_spread) for common_spread in common_spreads]

        return_df = pd.concat(return_dfs, keys=new_columns, axis=1)

        return return_df

    def _drop_df_columns(self, df: pd.DataFrame, level: int = 2) -> pd.DataFrame:
        """

        Args:
          df(pd.DataFrame): Input data frame
          level(int): Level on which to drop (Default value = 2)

        Returns:
            pd.DataFrame: Data frame with dropped columns

        """
        df_columns = np.array(df.columns.get_level_values(2).unique())
        columns_to_drop_index = np.isin(df_columns, list(self.feature_names_to_drop))
        columns_to_drop = df_columns[columns_to_drop_index]
        dropped_df = df.drop(columns_to_drop, axis=1, level=level)
        return dropped_df

    @staticmethod
    def _check_if_applicable(pair_names_1: List[str], pair_names_2: List[str]) -> bool:
        """

        Args:
            pair_names_1(List[str]): Names of pair 1
            pair_names_2(List[str]): Names of pair 2

        Returns:
            bool: Is operation possible or not

        """
        return pair_names_1[-1] == pair_names_2[-1]

    @staticmethod
    def _get_new_pair_name(pair_names_1: List[str], pair_names_2: List[str], pair_name_1_delimiter: str) -> str:
        """

        Args:
          pair_names_1(List[str]): Names of pair 1
          pair_names_2(List[str]): Names of pair 2
          pair_name_1_delimiter(str): Name delimiter

        Returns:
            str: New name after operation

        """
        return pair_names_1[0] + pair_name_1_delimiter + pair_names_2[0]

    def __call__(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
        """
        Dividing by levels in case of replacing level names. First drop self.feature_names_to_drop

        Args:
            df_1(pd.DataFrame): df_1 to be divided
            df_2(pd.DataFrame): df_2 to be divisor
        Returns:
            pd.DataFrame: Output data frame

        """

        """
        pair_name_1, pair_names_1, pair_name_1_delimiter = _get_pair_name_tuple(df_1)
        pair_name_2, pair_names_2, _ = _get_pair_name_tuple(df_2)

        assert self._check_if_applicable(pair_names_1, pair_names_2)  # USD_EUR x CAD_EUR
        new_pair_name = self._get_new_pair_name(pair_names_1, pair_names_2, pair_name_1_delimiter)  # USD_CAD

        dropped_df_1 = self._drop_df_columns(df_1)
        dropped_df_2 = self._drop_df_columns(df_2)

        multiplied_df = self.divide_level_1(dropped_df_1[pair_name_1], dropped_df_2[pair_name_2])
        """
        invert_transform = Invert()
        multiply_transform = Multiply()
        return_df = multiply_transform(df_1, invert_transform(df_2))

        return return_df

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

