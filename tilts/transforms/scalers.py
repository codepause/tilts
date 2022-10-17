import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler as StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
import inspect


class Snapshot:
    def __init__(self):
        self._snapshot = True

    # SNAPSHOT: its values are determined at the init and can data leak.
    # I.e. OpenScaler does not snapshot. (Not looking forward to calculate any values).
    # But StandadScaler does snapshot (possible data leak, so we have to divide data properly)

    @property
    def snapshot(self) -> bool:
        return self._snapshot


class CustomScaler(TransformerMixin, BaseEstimator, Snapshot):
    # Scaler for wrapping sklearn scalers to be able to work with DataFrames
    def __init__(self, scaler=None, behaviour='sklearn'):
        """
        Args:
            scaler:
            behaviour(str): select 'sklearn' or 'grouped' for different behaviour on grouped features' Grouped is equal to
                fit on flatten array.
        """
        super().__init__()
        self._scaler = scaler
        self._behaviour = behaviour
        self._initial_shape = None

    @property
    def behaviour(self) -> str:
        return self._behaviour

    @property
    def scaler(self):
        return self._scaler

    def _get_values(self, df: pd.DataFrame):
        if self._behaviour == 'grouped':
            X = df.values.reshape((-1, 1))
        else:
            X = df.values
        return X

    def _wrap(self, fnc):
        sig = inspect.signature(fnc)
        available_names = [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]
        def _finp(*args, **kwargs):
            df, *a = args
            kwgs = dict(kwargs)
            for name in kwargs:
                if name not in available_names:
                    kwgs.pop(name)
            init_shape = df.values.shape
            x = self._get_values(df)
            if fnc.__name__ == 'fit':
                return fnc(x, *a, **kwgs)
            else:
                x_fnc = fnc(x, *a, **kwgs)
                return pd.DataFrame(x_fnc.reshape(init_shape), index=df.index, columns=df.columns)

        return _finp

    def __getattr__(self, item):
        getattr(self._scaler, item)
        # return self._scaler.__getattribute__(item)

    def __getattribute__(self, item):
        if item in ['fit', 'fit_transform', 'inverse_transform', 'transform']:
            return self._wrap(self._scaler.__getattribute__(item))
        return super().__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return CustomScaler(scaler=self._scaler(), behaviour=self._behaviour)


class OpenScaler:
    def __init__(self):
        self.divided_by = None

    # Scaler to scale by Open
    def fit(self, df: pd.DataFrame, *args, **kwargs):
        if 'Open' in df.columns.get_level_values(2).unique():
            name_to_select = (kwargs.get('pair_name'), kwargs.get('spread_name'), 'Open')
            self.divided_by = df[name_to_select]

    def transform(self, df: pd.DataFrame, *args, **kwargs):
        if 'Open' in df.columns.get_level_values(2).unique():
            name_to_select = (kwargs.get('pair_name'), kwargs.get('spread_name'), 'Open')
            df = df.div(df[name_to_select], axis=0)
            df.drop('Open', axis=1, level=2, inplace=True)
        return df

    def inverse_transform(self, df: pd.DataFrame, divided_by: pd.DataFrame = None, **kwargs):
        if divided_by is not None:  # in real inference we dont have self.divided_by fitted
            name_to_select = (kwargs.get('pair_name'), kwargs.get('spread_name'), 'Open')
            divided_by = divided_by[name_to_select]
        elif self.divided_by is not None:
            divided_by = self.divided_by
        df = df.multiply(divided_by, axis=0).loc[df.index]
        return df

    def __call__(self, *args, **kwargs):
        return self


class MultiScaler:
    def __init__(self, name=None, scaler_mapping=None, default_scaler=None):
        """
        Args:
            scaler_mapping(dict) if we want different spreads to scale by different scaler types
                should pass dictionary of mapping like
                {'USD_EUR':{'mid':Standard,'ask':MinMax}} or just {'mid':Standard,'ask':MinMax}
            default_scaler: if no mappings for name use it. Mappings could contain None as no scaler to apply

        """
        self.name = name
        if scaler_mapping is None:
            scaler_mapping = {}
        assert scaler_mapping is not None or default_scaler is not None

        self.scale_info = dict()
        self.default_scaler = default_scaler
        self.scaler_mapping = scaler_mapping
        self.scaler_mapping_depth = self.__get_scaler_mapping_depth()

    def __get_scaler_mapping_depth(self) -> int:
        """
        depth 2 means we have scaler mapping for every pair_name
        depth 1 - sor spread_level
        depth 0 - always use default
        https://stackoverflow.com/questions/23499017/know-the-depth-of-a-dictionary

        Returns:
            int: Depth of mapping dictionary
        """
        from collections import deque

        def depth(d):
            """

            Args:
              d: 

            Returns:

            """
            queue = deque([(id(d), d, 1)])
            memo = set()
            while queue:
                id_, o, level = queue.popleft()
                if id_ in memo:
                    continue
                memo.add(id_)
                if isinstance(o, dict):
                    queue += ((id(v), v, level + 1) for v in o.values())
            return level

        return depth(self.scaler_mapping)

    def _fit_on_features(self, df: pd.DataFrame, pair_name: str, spread_name: str, is_train=False, **kwargs):
        """
        Spread level DataFrame only (containing 1 level)

        Args:
          df(pd.DataFrame): Input data frame
          pair_name(str): Pair name to fit on
          spread_name(str): Spread name to fit on

        Returns:
            Union[None, 'scaler']: Fitted scaler on features

        """
        if self.scaler_mapping_depth == 2:
            scaler_instance = self.scaler_mapping.get(spread_name, None)
        elif self.scaler_mapping_depth == 3:
            temp = self.scaler_mapping.get(pair_name, dict())
            scaler_instance = temp.get(spread_name, None)
        else:
            scaler_instance = self.default_scaler

        if scaler_instance is not None:
            if is_train or not isinstance(scaler_instance, Snapshot):
                scaler = scaler_instance()
                # print('fitting', scaler, 'on', pair_name, spread_name)
                scaler.fit(df, pair_name=pair_name, spread_name=spread_name, **kwargs)
            else:
                scaler = self.scale_info[pair_name][spread_name]['scaler']
        else:
            scaler = None

        return scaler

    def fit(self, df: pd.DataFrame, is_train=False, **kwargs):
        """
        Fit created scalers

        Args:
          df(pd.DataFrame): Data frame with 3 levels
          is_train(bool): If to create new scaler from instance when fitting

        """
        pair_names = df.columns.get_level_values(0).unique()
        for pair_name in pair_names:
            spread_names = df[pair_name].columns.get_level_values(0).unique()
            self.scale_info[pair_name] = self.scale_info.get(pair_name, dict())
            for spread_name in spread_names:
                self.scale_info[pair_name][spread_name] = self.scale_info[pair_name].get(spread_name, dict())
                feature_df = df.xs((pair_name, spread_name), axis=1, drop_level=False, level=0)
                feature_names = feature_df.columns.get_level_values(0).unique()
                self.scale_info[pair_name][spread_name]['feature_names'] = list(feature_names)
                self.scale_info[pair_name][spread_name]['scaler'] = self._fit_on_features(feature_df, pair_name,
                                                                                          spread_name,
                                                                                          is_train=is_train)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns scaled copy of df

        Args:
          df(pd.DataFrame): Data frame with 3 levels

        Returns:
            pd.DataFrame: Scaled data frame

        """
        pair_names = df.columns.get_level_values(0).unique()
        pair_dfs = []
        for pair_name in pair_names:
            spread_names = df[pair_name].columns.get_level_values(0).unique()
            spread_dfs = []
            for spread_name in spread_names:
                scaler = self.scale_info[pair_name][spread_name]['scaler']
                feature_df = df.xs((pair_name, spread_name), axis=1, drop_level=False, level=0)
                feature_df_columns = feature_df.columns.get_level_values(0)
                if scaler is not None:
                    scaled_feature_df = scaler.transform(feature_df, pair_name=pair_name, spread_name=spread_name,
                                                         **kwargs)
                    # scaled_feature_df = pd.DataFrame(scaled_feature_values, columns=feature_df_columns,
                    #                                  index=feature_df.index)
                else:
                    scaled_feature_df = feature_df
                pair_dfs.append(scaled_feature_df)
            # spread_df = pd.concat(spread_dfs, keys=spread_names, axis=1)
            # pair_dfs.append(spread_df)
        return_df = pd.concat(pair_dfs, axis=1)
        return return_df

    def fit_transform(self, df: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        """

        Args:
            df(pd.DataFrame): Data frame with 3 levels
            is_train(bool): if is training. Decides to fit or not for snapshot-type scalers

        Returns:
            pd.DataFrame: Fitted and transformed data frame
        """
        self.fit(df, is_train=is_train, **kwargs)
        return self.transform(df)

    def save(self, path: str):
        """
        Saving current class

        Args:
          path(str): Save path

        """
        assert os.path.exists(os.path.dirname(path))
        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pair_names = df.columns.get_level_values(0).unique()
        pair_dfs = []
        for pair_name in pair_names:
            spread_names = df[pair_name].columns.get_level_values(0).unique()
            spread_dfs = []
            for spread_name in spread_names:
                scaler = self.scale_info[pair_name].get(spread_name, {}).get('scaler', None)
                feature_df = df.xs((pair_name, spread_name), axis=1, drop_level=False, level=0)
                if scaler is not None:
                    scaled_feature_df = scaler.inverse_transform(feature_df, pair_name=pair_name,
                                                                 spread_name=spread_name, **kwargs)
                    # scaled_feature_df = pd.DataFrame(scaled_feature_values, columns=feature_df_columns,
                    #                                  index=feature_df.index)
                else:
                    scaled_feature_df = feature_df
                pair_dfs.append(scaled_feature_df)
            # spread_df = pd.concat(spread_dfs, keys=spread_names, axis=1)
            # pair_dfs.append(spread_df)
        return_df = pd.concat(pair_dfs, axis=1)
        return return_df

    @staticmethod
    def load(path: str) -> 'MultiScaler':
        """

        Args:
          path(str): Load path

        Returns:
            'MultiScaler': Loaded 'MultiScaler'

        """
        assert os.path.exists(os.path.dirname(path))
        with open(path, 'rb') as f:
            multi_scaler = pickle.load(f)
        return multi_scaler

    def __repr__(self):
        return self.__class__.__name__ + f'({self.scaler_mapping})'


if __name__ == '__main__':
    import numpy as np

    df = pd.DataFrame(np.random.random((10, 3)),
                      columns=pd.MultiIndex.from_tuples(
                          [('USD_EUR', 'mid', 'Open'), ('USD_EUR', 'mid', 'Low'), ('USD_EUR', 'mid', 'Close')]))

    scaler = MultiScaler({'mid': CustomScaler(scaler=StandardScaler, behaviour='grouped')})
    # scaler = MultiScaler({'mid': OpenScaler})
    scaled_df = scaler.fit_transform(df)
