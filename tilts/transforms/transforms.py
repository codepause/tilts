from typing import List, Union, Dict

import numpy as np
import pandas as pd
from collections import defaultdict
from .utils import *

try:
    import tsfresh
    from tsfresh import extract_features, extract_relevant_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.utilities.dataframe_functions import roll_time_series
    from tsfresh import extract_features, select_features
except ImportError:
    import logging

    logging.warning('Tsfresh transform is not available due package is not installed')

from epta.core.tool import BaseTool
import epta.core.base_ops as ecb


class DfTool(BaseTool):
    """ Applicable only to dfs """

    def __init__(self, name: str = 'DfTool', **kwargs):
        super(DfTool, self).__init__(name=name, **kwargs)

    def __call__(self, df: pd.DataFrame, **kwargs):
        assert isinstance(df, pd.DataFrame)
        return self.use(df, **kwargs)


class Compose(ecb.Sequential):
    pass


class Invert(DfTool):
    """Inverts pandas df USD_EUR to EUR_USD with respect to ask/bid and high/low"""
    names_to_invert: Tuple[str]

    def __init__(self, spread_names_to_invert: Tuple[str] = ('ask', 'bid', 'mid'),
                 feature_names_to_invert: Tuple[str] = ('Open', 'High', 'Low', 'Close'),
                 **kwargs):
        self.feature_names_to_invert = feature_names_to_invert
        self.spread_names_to_invert = spread_names_to_invert
        super(Invert, self).__init__(name='Invert', **kwargs)

    def invert_level_2(self, df_level_2: pd.DataFrame) -> pd.DataFrame:
        """
        Deals with invert level 2 values (O H L C)

        Args:
            df_level_2(pd.DataFrame): Data frame on level 2 of main df

        Returns:
            pd.DataFrame: Data frame with inverted level 2 values (inverted column names and values)

        """
        feature_level: list = df_level_2.columns.unique()
        feature_level_replace = {'High': 'Low', 'Low': 'High'}

        invert_indicator = np.isin(feature_level, self.feature_names_to_invert)

        inverted_df = df_level_2.loc[:, invert_indicator].rdiv(1)
        straight_df = df_level_2.loc[:, ~invert_indicator]

        return_df = pd.concat([inverted_df, straight_df], axis=1)

        new_columns = [feature_level_replace.get(column_name, column_name) for column_name in return_df.columns]
        return_df.columns = new_columns

        return return_df.reindex(feature_level, axis=1)

    def invert_level_1(self, df_level_1: pd.DataFrame) -> pd.DataFrame:
        """
        Deals with invert level 1 values (ask, bid). Uses invert_level2

        Args:
            df_level_1: pd.DataFrame that is on level 1 of main df

        Returns:
            pd.DataFrame: Data frame with inverted level 2 and level 1

        """
        spread_level: list = df_level_1.columns.get_level_values(0).unique()
        spread_level_replace = {'ask': 'bid', 'bid': 'ask'}

        invert_indicator = np.isin(spread_level, self.spread_names_to_invert)

        return_dfs = [self.invert_level_2(df_level_1[df_name]) if invert_indicator[idx] else df_level_1[df_name] for
                      idx, df_name in enumerate(spread_level)]
        new_columns = [spread_level_replace.get(column_name, column_name) for column_name in spread_level]

        return_df = pd.concat(return_dfs, keys=new_columns, axis=1)

        return return_df.reindex(spread_level, axis=1, level=0)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Inverted data frame

        """
        pair_name: str = df.columns.get_level_values(0).unique()[0]

        # find pair name delimiter and invert name
        pair_name_delimiter = get_pair_name_delimiter(pair_name)
        new_pair_name = pair_name_delimiter.join(pair_name.split(pair_name_delimiter)[::-1])

        inverted_df = self.invert_level_1(df[pair_name])

        # adding top level
        named_inverted_df = pd.concat([inverted_df], keys=[new_pair_name], axis=1)

        return named_inverted_df


class ToMid(DfTool):
    """Makes mid from ask and bid"""

    def __init__(self, *args, name: str = 'ToMid', **kwargs):
        super(ToMid, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Mid data frame

        """
        # pair_name: str = df.columns.get_level_values(0).unique()[0]
        spread_level_names = df.columns.get_level_values(1).unique()

        frame_dummy = list()
        for pair_name in df.columns.get_level_values(0).unique():
            mid_df = (df[pair_name]['ask'] + df[pair_name]['bid']) / 2
            mid_df = pd.concat([mid_df], keys=['mid'], axis=1)
            if 'volume' in spread_level_names:
                mid_df = pd.concat([mid_df, df[pair_name][['volume']]], axis=1)
            mid_df = mid_df.dropna(axis=1, how='all')  # in case some features does not exist in both spreads
            mid_df = pd.concat([mid_df], keys=[pair_name], axis=1)
            frame_dummy.append(mid_df)

        return_df = pd.concat(frame_dummy, axis=1)
        return return_df


class ToDaily(DfTool):
    """Makes days from hours. Works on mid only currently"""

    def __init__(self, day_len: int, group_by: list = ['year', 'month', 'day'], verbose=False, name: str = 'ToDaily',
                 **kwargs):
        self.day_len = day_len
        self.group_by = group_by  # how to group data by time index. like ['date', 'hour', 'minute']
        self.verbose = verbose

        self.__apply_group = {'mid', 'ask', 'bid', 'volume'}
        super(ToDaily, self).__init__(name=name, **kwargs)

    def __filter_day(self, df: pd.DataFrame, key: str) -> Union[pd.DataFrame, None]:
        if key == 'Open':
            return df[key].iloc[0]
        elif key == 'High':
            return np.amax(df[key])
        elif key == 'Low':
            return np.amin(df[key])
        elif key == 'Close':
            return df[key].iloc[-1]
        elif key == 'Volume':
            return np.sum(df[key])
        else:
            return None

    def _filter_data(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df_group(pd.DataFrame): Data frame from groupby func

        Returns:
            pd.DataFrame: Filtered data frame with use of some operations

        """
        timestamp_data = {key: getattr(df_group.index[0], key) for key in self.group_by}
        timestamp_data.update({'tz': getattr(df_group.index[0], 'tz')})
        new_index = [pd.Timestamp(**timestamp_data)]
        # if 'volume' in df_group.columns.get_level_values(1).unique():
        #     df_group = df_group.drop('volume', axis=1, level=1)

        frame_dummy = dict()
        for pair_name in df_group.columns.get_level_values(0).unique():
            for spread in df_group.columns.get_level_values(1).unique():
                if spread in self.__apply_group:
                    for key in df_group[pair_name][spread]:
                        value = self.__filter_day(df_group[pair_name][spread], key)
                        if value:
                            frame_dummy[(pair_name, spread, key)] = self.__filter_day(df_group[pair_name][spread], key)
                else:
                    frame_dummy[(pair_name, spread)] = df_group[pair_name][spread]

        total_hours = len(df_group)

        if total_hours < self.day_len:
            return_df = pd.DataFrame(columns=df_group.columns)
        else:
            return_df = pd.DataFrame(frame_dummy, columns=df_group.columns, index=new_index)
        return return_df

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Daily data frame

        """
        times = df.index
        gr_param = [getattr(times, param) for param in self.group_by]
        grpb = df.groupby(gr_param, group_keys=False)
        grouped_data = grpb.apply(lambda df_group: self._filter_data(df_group))
        return grouped_data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.day_len})'


class Concat(ecb.Concatenate, DfTool):
    """Apply and concatenate transforms not as a composition but in parallel"""

    def __init__(self, tools: list, remove_duplicated_columns=True, name: str = 'Concat', **kwargs):
        self.remove_duplicated_columns = remove_duplicated_columns
        super(Concat, self).__init__(tools=tools, name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Concatenated transformed data frames of the initial data frame

        """
        transformed_dfs = super(Concat, self).use(df)

        concatenated_df = pd.concat(transformed_dfs, axis=1)
        if self.remove_duplicated_columns:
            duplicated_index = concatenated_df.columns.duplicated()
            concatenated_df = concatenated_df.loc[:, ~duplicated_index]
        return concatenated_df

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Eye(DfTool):
    """Skip the transformation. Just return df. Used in concat, to also return starting data"""

    def __init__(self, name: str = 'Eye', **kwargs):
        super(Eye, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Input data frame

        """
        return df


class DropColumns(DfTool):
    """Drop columns"""

    def __init__(self, columns: List[Tuple[str, int]], name: str = 'DropColumns', **kwargs):
        self.columns_to_drop = columns
        super(DropColumns, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: df with dropped columns

        """
        for column_name, column_level in self.columns_to_drop:
            df = df.drop(column_name, axis=1, level=column_level)
        df.columns.remove_unused_levels()
        return df


class DatetimeIndexFilter(DfTool):
    """Filter index with given data for year/month etc"""

    def __init__(self, name: str = 'DatetimeIndexFilter', **kwargs):
        self.kwargs = kwargs
        self.sets = {k: set(v) for k, v in kwargs.items()}
        super(DatetimeIndexFilter, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: filtered df

        """
        mask = df.index.map(lambda x: np.all(
            [x.__getattribute__(attr_name) in attr_values for attr_name, attr_values in self.sets.items()])) == True
        return df[mask]

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.kwargs})'


class Lambda(ecb.Lambda):
    """Apply your custom transformation with lambda! Use Lambda and be happy!"""

    def __init__(self, *args, **kwargs):
        super(Lambda, self).__init__(*args, **kwargs)


class Tfresh(DfTool):
    def __init__(self, settings: dict = None, min_timeshift: int = 5, max_timeshift: int = 20, target: str = 'High',
                 name: str = 'Tfresh', **kwargs):
        self._min_timeshift = min_timeshift
        self._max_timeshift = max_timeshift
        self._target = target
        if settings:
            self._settings = settings
        else:
            self._settings = defaultdict(dict)
        super(Tfresh, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # https://tsfresh.readthedocs.io/en/latest/index.html
        # TODO: speedup this by grouping features.
        assert len(df) > self._min_timeshift
        ticker_name = df.columns.get_level_values(0).unique()[0]
        ticker_features = list()
        if ticker_name in self._settings:
            for spread_name in df.columns.get_level_values(1).unique():
                spread_features = list()
                for feature_name in df.columns.get_level_values(2).unique():
                    if feature_name in self._settings[ticker_name].get(spread_name, dict()):
                        settings = self._settings[ticker_name][spread_name][feature_name]
                        feature_df = df[ticker_name][spread_name][[feature_name]]
                        features_filtered = self.transform(feature_df, settings=settings)
                        features_filtered = pd.concat([features_filtered], axis=1, keys=[spread_name + '_tfresh'])
                        features_filtered.set_index(features_filtered.index.get_level_values(1), inplace=True)
                        spread_features.append(features_filtered)
                spread_features_df = pd.concat(spread_features, axis=1) if spread_features else pd.DataFrame(
                    columns=df.columns)
                ticker_features.append(spread_features_df)
        else:
            for spread_name in df.columns.get_level_values(1).unique():
                ticker_df = df[ticker_name]
                features_filtered = self.fit_transform(ticker_df, spread_name)
                self._settings[ticker_name][spread_name] = tsfresh.feature_extraction.settings.from_columns(
                    list(features_filtered.columns))
                features_filtered = pd.concat([features_filtered], axis=1, keys=[spread_name + '_tfresh'])
                features_filtered.set_index(features_filtered.index.get_level_values(1), inplace=True)
                ticker_features.append(features_filtered)
        ticker_features_df = pd.concat(ticker_features, axis=1, keys=[ticker_name])
        ticker_features_df = ticker_features_df.reindex(sorted(ticker_features_df.columns), axis=1)
        return ticker_features_df

    @staticmethod
    def fit(df_features: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        features_filtered = select_features(df_features, y)
        return features_filtered

    def transform(self, spread_df: pd.DataFrame, settings: dict = None) -> pd.DataFrame:
        spread_df.loc[:, 'time'] = spread_df.index
        spread_df.loc[:, 'id'] = np.ones(len(spread_df.index))
        # create windows in data
        df_rolled = roll_time_series(spread_df, column_id="id", column_sort="time",
                                     min_timeshift=self._min_timeshift, max_timeshift=self._max_timeshift)
        # for every window created we assume target as next day high value
        df_features = extract_features(df_rolled, settings, column_id="id", column_sort="time", impute_function=impute,
                                       chunksize=1)
        return df_features

    def fit_transform(self, ticker_df: pd.DataFrame, spread_name: str) -> pd.DataFrame:
        spread_df = pd.DataFrame(ticker_df[spread_name])
        df_features = self.transform(spread_df)
        y = ticker_df.xs(self._target, axis=1, level=1, drop_level=False)  # target is one value
        y = y.iloc[self._min_timeshift:].shift(-1).set_index(df_features.index).fillna(
            method='ffill')
        y = y.squeeze(axis=1)  # to series
        features_filtered = self.fit(df_features, y)
        return features_filtered


class ApplyToEach(DfTool):
    """ Apply transform to every column in multiindex """

    def __init__(self, transforms: Union[
        List[DfTool], Dict[str, DfTool], DfTool, Dict[str, List[DfTool]]] = None, name: str = 'ApplyToEach', **kwargs):
        if not transforms:
            self.transforms = Eye()
        else:
            self.transforms = transforms
        super(ApplyToEach, self).__init__(name=name, **kwargs)

    def _get_transform(self, ticker_name: str) -> DfTool:
        if isinstance(self.transforms, list):
            return Concat(self.transforms)  # to match data_loader
        elif isinstance(self.transforms, DfTool):
            return self.transforms
        elif isinstance(self.transforms, dict):
            t = self.transforms.get(ticker_name, Eye())
            # convert list to transform
            if isinstance(t, list):
                return Concat(t)
            else:
                return t

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = list()
        for ticker_name in df.columns.get_level_values(0).unique():
            transform = self._get_transform(ticker_name)
            res = transform(df[[ticker_name]])
            ret.append(res)
        ret = pd.concat(ret, axis=1)
        return ret


class Dropna(DfTool):
    def __init__(self, *args, name: str = 'Dropna', **kwargs):
        self.args = args
        self.kwargs = kwargs
        super(Dropna, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.dropna(*self.args, **self.kwargs)


class Loc(DfTool):
    """Slice data by timestamp"""

    def __init__(self, start: pd.Timestamp = None, end: pd.Timestamp = None, name: str = 'TakeFromDate',
                 **kwargs):
        self.start = start
        self.end = end
        super(Loc, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: sliced df by timestamp (loc)

        """
        return df.loc[self.start:self.end]

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.start}, {self.end})'


class Iloc(DfTool):
    def __init__(self, start: int = None, end: int = None, name: str = 'Slice', **kwargs):
        self.start = start
        self.end = end
        super(Iloc, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.iloc[self.start:self.end]


class AsType(DfTool):
    def __init__(self, dtype: type, name: str = 'AsType', **kwargs):
        self.dtype = dtype
        super(AsType, self).__init__(name=name, **kwargs)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.astype(self.dtype)


class GroupNews(DfTool):
    """ Groups dataframe consisted of lists"""

    def __init__(self, granularity: str, name: str = 'GroupNews', **kwargs):
        self.granularity = granularity
        super(GroupNews, self).__init__(name=name, **kwargs)

    def _filter_data(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df_group(pd.DataFrame): Data frame from groupby func

        Returns:
            pd.DataFrame: Filtered data frame with use of some operations

        """
        if df_group.empty:
            return pd.DataFrame()
        frame_dummy = dict()
        for column_name in df_group.columns:
            frame_dummy[column_name] = [df_group[column_name].tolist()]

        return_df = pd.DataFrame(frame_dummy)
        return return_df

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Daily data frame

        """

        grouped_data = df.groupby(pd.Grouper(freq=self.granularity)).apply(
            lambda df_group: self._filter_data(df_group))
        grouped_data.reset_index(level=1, drop=True, inplace=True)
        return grouped_data

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class FlattenNews(DfTool):
    """Make dummy dataframe from list of features"""

    def __init__(self, max_pad: str, name: str = 'FlattenNews', **kwargs):
        self.max_pad = max_pad
        super(FlattenNews, self).__init__(name=name, **kwargs)

    def _apply(self, df: pd.DataFrame) -> np.array:
        if not df:
            return df
        if isinstance(df[0], (np.ndarray, np.generic)):
            vals = np.array(df)
            dummy_array = np.zeros((self.max_pad * vals.shape[-1]))
            dummy_array[:min(vals.size, self.max_pad * vals.shape[-1])] = vals.flatten()[:self.max_pad * vals.shape[-1]]
            return dummy_array
        else:
            return df

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Daily data frame
        """

        frame_dummy = dict()
        features_dummy = list()
        for column_name in df.columns:
            if 'features' in column_name:
                res = df[column_name].apply(lambda x: self._apply(x))
                temp_df = pd.DataFrame(res.to_list(), index=res.index)
                features_dummy.append(pd.concat([temp_df], keys=[column_name[:2]], axis=1))
            else:
                frame_dummy[column_name] = df[column_name].apply(lambda x: self._apply(x))

        features_df = pd.concat(features_dummy, axis=1)
        text_df = pd.DataFrame(frame_dummy)
        return_df = pd.concat([features_df, text_df], axis=1)
        return return_df

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ClockLoc(DfTool):
    def __init__(self, clock: 'Clock', name: str = 'ClockLoc', **kwargs):
        self.clock = clock
        super(ClockLoc, self).__init__(name=name, **kwargs)

    def __call__(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Daily data frame
        """

        start_key, end_key, end_key_index = get_time_from_n_entries(df, df.index, self.clock, kwargs.get('n_entries'),
                                                                    granularity=kwargs.get('granularity'))
        sliced_return_df = df.loc[start_key:end_key]
        return sliced_return_df


class PdIndexLexsort(DfTool):
    def __init__(self, name: str = 'PdIndexLexsort', **default_kwargs):
        self.default_kwargs = default_kwargs
        super(PdIndexLexsort, self).__init__(name=name)

    def use(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: lexsort for faster indexing
        """
        kwgs = dict(self.default_kwargs)
        kwgs.update(kwargs)
        return df.sort_index(**kwgs)


class TAlib(DfTool):
    """TAlib wrapper for pickle support"""

    def __init__(self, fnc, args, params, name: str = 'TAlib', **kwargs):
        self.fnc = fnc
        self.args = args
        self.params = params
        super(TAlib, self).__init__(name=name, **kwargs)

    def use(self, df, **kwargs) -> pd.DataFrame:
        """
        Args:
            df(pd.DataFrame): Input data frame

        Returns:
            pd.DataFrame: Talib fnc with *args, **kwargs inited
        """
        ticker_name = df.columns.get_level_values(0).unique()[0]
        args_data = [df[ticker_name]['mid'][i] for i in self.args]
        # import pdb
        # pdb.set_trace()
        new_df = self.fnc(*args_data, **self.params)  # return pd dataframe with multiple columns its k
        new_df = pd.DataFrame(new_df, index=df.index)  # if index is not fully consumed b4
        new_df.columns = pd.MultiIndex.from_product(
            [[ticker_name], [self.fnc.__name__ + '_' + '_'.join(self.args)], new_df.columns])
        return new_df


if __name__ == '__main__':
    pass
