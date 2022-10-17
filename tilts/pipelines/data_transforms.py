import pandas as pd

from tilts.transforms.scalers import MultiScaler
from tilts.pipelines.base_pipeline import Pipeline


class DataTransforms(Pipeline):
    def __init__(self, pipeline: list):
        self.pipeline = pipeline
        self.scalers = list()

    def _call_item(self, item, data: pd.DataFrame, is_train: bool = False):
        if isinstance(item, MultiScaler):
            data = item.fit_transform(data, is_train=is_train)
            if item not in self.scalers:
                self.scalers.append(item)
        else:
            data = item(data)
        return data

    def __call__(self, pairs_data: pd.DataFrame, *args, **kwargs):
        res = pairs_data
        is_train = kwargs.get('is_train', False)
        for item in self.pipeline:
            res = self._call_item(item, res, is_train)
        return res  # {'pairs_data': res, 'data_scalers': self.scalers}

    def __repr__(self):
        st = f''
        for item in self.pipeline:
            st += str(item) + '\n'
        return st
