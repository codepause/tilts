import pandas as pd
from typing import Union, Tuple

from tilts.datasets.meta import FlatCriterion
from tilts.datasets.meta import Block


class WindowCriterion(FlatCriterion):
    def __init__(self, delta_start: Union[pd.Timedelta, int, None], delta_end: Union[pd.Timedelta, int, None],
                 rounds: Tuple[str, str] = (None, None)):
        """
        :param delta_start: relative timedelta to block idx
        :param delta_end: relative timedelta to block idx
        :param rounds: tuple for start and end to round: (hours_up, hours_down) ...
        """
        self._delta_start = delta_start
        self._delta_end = delta_end

        self._rounds = list()
        for round_ in rounds:
            if isinstance(round_, str):
                self._rounds.append(round_.split('_'))
            else:
                self._rounds.append((None, None))

        # year should not be used
        self._resolutions = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
        self._resolutions_idx_mappings = dict(zip(self._resolutions, range(len(self._resolutions))))

    def apply(self, block: 'Block', **kwargs) -> 'Block':
        if isinstance(self._delta_end, int):
            end_key_idx = block.idx + self._delta_end
            if end_key_idx > len(block.data.index) - 1 or end_key_idx < 0:
                end_key = None
            else:
                end_key = block.data.index[end_key_idx]
        elif isinstance(self._delta_end, pd.Timedelta):
            end_key = block.timestamp_idx + self._delta_end
        else:
            end_key = None

        if isinstance(self._delta_start, int):
            start_key_idx = block.idx + self._delta_start
            if start_key_idx < 0 or start_key_idx > len(block.data.index) - 1:
                start_key = None
            else:
                start_key = block.data.index[start_key_idx]
        elif isinstance(self._delta_start, pd.Timedelta):
            start_key = block.timestamp_idx + self._delta_start
        else:
            start_key = None

        # to exclude the situation both indices out of array
        if start_key is None and end_key is None:
            return block.spawn(None)

        new_keys = list()
        for idx, time_key in enumerate([start_key, end_key]):
            round_, to = self._rounds[idx]
            new_key = time_key
            if round_:  # 'hour' means that everything starting from hour and less will be set to 0
                replace_kwargs = {}
                for resolution in self._resolutions[self._resolutions_idx_mappings[round_]:]:
                    replace_kwargs[resolution[:-1]] = 0
                if to == 'up':
                    round_delta = self._resolutions[self._resolutions_idx_mappings[round_] - 1]
                    new_key = time_key.replace(**replace_kwargs) + pd.Timedelta(**{round_delta: 1}) - pd.Timedelta(
                        seconds=1)
                elif to == 'down':
                    new_key = time_key.replace(**replace_kwargs)
            new_keys.append(new_key)
        start_key, end_key = new_keys
        new_data = block.data.loc[start_key:end_key]

        if new_data.empty:
            return block.spawn(None)
        else:
            return block.spawn(new_data)
