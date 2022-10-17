from typing import Tuple, Any, Optional
import pandas as pd
import re


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


# No pickle support coz locals. not wrappable by lambda in that case. will be depreciated
def _talib(fnc: callable, args: list, params: dict):
    def transform(df: pd.DataFrame):
        ticker_name = df.columns.get_level_values(0).unique()[0]
        args_data = [df[ticker_name]['mid'][i] for i in args]
        new_df = fnc(*args_data, **params)  # return pd dataframe with multiple columns its k
        new_df = pd.DataFrame(new_df, index=df.index)  # if index is not fully consumed b4
        new_df.columns = pd.MultiIndex.from_product(
            [[ticker_name], [fnc.__name__], [fnc.__name__ + '_' + '_'.join(args)]])
        return new_df

    return transform


def __get_kwargs(granularity: str, n: int):
    # get right amount of requesting dates
    mappings = {'h': 'hours', 'd': 'days', 'm': 'minutes', 's': 'seconds'}
    val = int(granularity[0])
    val = val * n  # add 5 for safe request
    key = granularity[-1].lower()
    return {mappings[key]: val}


def get_time_from_n_entries(df: pd.DataFrame, df_index: pd.Index, clock: 'Clock', n, granularity=None,
                            **kwargs) -> Tuple[
    Optional[Any], Any, Any]:
    # get last n entries if n is int or get timedelta if timedelta

    end_key = clock.current_time
    end_key_index = None
    if n is not None:
        if granularity and end_key is not None:
            # add little timedelta for proper slicing. Ie. if n == 1, only one entity should be returned
            start_key = end_key - pd.Timedelta(**__get_kwargs(granularity, n)) + pd.Timedelta(milliseconds=1)
        else:
            if end_key is None:
                end_key = df_index[-1]
            elif end_key < df_index[0]:
                # what if the key is before all available data?
                start_key = end_key
                return start_key, end_key, end_key_index
            if isinstance(n, int):
                end_key_index = df_index.get_loc(end_key, 'ffill')
                start_key = df_index[max(end_key_index - n + 1, 0)]
            elif isinstance(n, pd.Timestamp):
                start_key = end_key - n
            else:
                start_key = None
    else:
        start_key = None
    return start_key, end_key, end_key_index
