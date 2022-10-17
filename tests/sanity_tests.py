from easydict import EasyDict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# https://github.com/twopirllc/pandas-ta
import pandas_ta as ta  # pandas wrapper for talib

import epta.core.base_ops as etb

from tilts.apis.single_file_csv_api import SingleFileCsvApi
from tilts.modules.data_loader import DataLoader

from tilts.transforms import transforms as dtr
from tilts.transforms import scalers as dsc

from tilts.datasets.block_builders.block_builder import BlockBuilder
from tilts.datasets.criterions import *

from tilts.pipelines.data_transforms import DataTransforms
from tilts.pipelines.block_augs import BlockAugs


def create_config():
    config = EasyDict()

    # Overall data setup
    config.pairs_to_load = []
    config.load_from_date = pd.Timestamp(year=2022, month=8, day=1, tz='GMT')

    # Data transforms
    config.min_timeshift = 5
    config.max_timeshift = 90
    config.features_target = 'Close'

    base_transform = dtr.Compose([
        dtr.TakeFromDate(date_from=config.load_from_date),
        # dtr.DatetimeIndexFilter(hour=range(11, 20)),  # 20 not included
        # dtr.ToDaily(4)
        # dtr.DropColumns([('Low', 2), ('High', 2)]),
    ])

    talib_transform_data = [
        (ta.macd, ['Close'], {}),  # {'fillna': False}),
        (ta.rsi, ['Close'], {}),  # {'fillna': False}),
        (ta.massi, ['High', 'Low'], {}),  # {'fillna': False}),
        # (ta.trend.ichimoku_b, ['High', 'Low'], {'fillna': False}),
        (ta.trix, ['Close'], {}),  # {'fillna': False})
    ]
    talib_transform = dtr.Compose([
        base_transform,
        dtr.Concat([dtr.TAlib(*j) for j in talib_transform_data]),
    ])
    tsfresh_transform = dtr.Compose([
        dtr.Concat([
            base_transform,
            talib_transform,
        ]),
        dtr.Dropna(),
        dtr.Tfresh(min_timeshift=config.min_timeshift, max_timeshift=config.max_timeshift,
                   target=config.features_target)
    ])

    all_transforms = dtr.Compose([
        dtr.Concat([
            base_transform,
            talib_transform,
            # tsfresh_transform
        ]),
        dtr.Dropna()
    ])

    # transforms to apply to each ticker
    config.base_transforms_per_ticker = dict(zip(config.pairs_to_load, [all_transforms for _ in config.pairs_to_load]))

    config.augments_per_ticker = dict()

    # create scaler that acts on multiindex dataframe
    common_scaler = dsc.MultiScaler(
        scaler_mapping={
            'mid': dsc.CustomScaler(scaler=StandardScaler, behaviour='grouped'),
            'volume': dsc.CustomScaler(scaler=StandardScaler)
        }
    )

    # config.train_to_date = pd.Timestamp('')
    config.val_samples = 5500

    # THIS PIPELINE IS APPLIED AFTER DATA WAS LOADED FROM API
    config.pretransform_pipeline = dtr.ApplyToEach(config.base_transforms_per_ticker)
    config.data_train_pipeline = DataTransforms([
        # dtr.ApplyToEach(cfg.base_transforms_per),  # because some operations are not on index
        dtr.Compose([
            dtr.Slice(start=config.max_timeshift),
            dtr.Dropna(), dtr.AsType('float32'),
            dtr.Slice(end=-config.val_samples)
        ]),
        dtr.PdIndexLexsort(axis=1),
        dsc.MultiScaler(name='OpenScaler', scaler_mapping={'mid': dsc.OpenScaler}),
        common_scaler,
    ])

    # THIS PIPELINE IS APPLIED EVERY DATASET __getitem__ CALL
    config.x_aug_train_pipeline = BlockAugs([])  # BlockAugs([NoiseAug(), dtr.AsType('float32')])
    config.y_aug_train_pipeline = BlockAugs([])

    config.data_val_pipeline = DataTransforms([
        # dtr.ApplyToEach(cfg.base_transforms_per),
        dtr.Compose([
            # dtr.Slice(start=-cfg.val_samples),
            dtr.Slice(start=-config.val_samples * 2),
            dtr.Dropna(),
            dtr.AsType('float32'),
            dtr.Slice(end=-config.val_samples)
        ]),
        dtr.PdIndexLexsort(axis=1),
        dsc.MultiScaler(name='OpenScaler', scaler_mapping={'mid': dsc.OpenScaler}),
        common_scaler,

    ])

    config.data_test_pipeline = DataTransforms([
        # dtr.ApplyToEach(cfg.base_transforms_per),
        dtr.Compose([
            dtr.Slice(start=-config.val_samples),
            dtr.Dropna(),
            dtr.AsType('float32')
        ]),
        dtr.PdIndexLexsort(axis=1),
        dsc.MultiScaler(name='OpenScaler', scaler_mapping={'mid': dsc.OpenScaler}),
        common_scaler,
    ])

    # Dataset building
    config.val_ratio = 0.15

    # THIS BUILDER IS APPLIED ONCE TO BUILD indices and cache data if inline is TRUE,
    # else applied every time item is being gotten from dataset.
    # transforming df to block inside. similar to aug pipeline

    # make sure x has -1 as last index when slicing to prevent data leakage in pips_test.
    # Assuming bot launching the day it predicts
    config.x_builder = BlockBuilder([
        WindowCriterion(-14, -1),
        SampleLenCriterion(min_len=14)
    ])

    config.y_builder = BlockBuilder([
        WindowCriterion(0, 0),
        XsCriterion([config.features_target], level=2),
        # Filter values from distribution
        LambdaCriterion(
            lambda block, **kwargs: block if block.data is not None and np.all(block.data.max() < 3) else block.spawn(
                None))
    ])

    config.dataset_inline = True
    return config


def get_api_data(cfg: EasyDict) -> dict:
    api = SingleFileCsvApi('../../tinkoff/dumps/market_data/5m/BBG004730N88/data.csv')
    print('csv loaded')
    loader = DataLoader(api)
    pairs_data = loader.get_pairs_data_with_transforms(cfg.pairs_to_load)
    # or pairs_data = api.get_pair_data() for raw data.

    transformed_pairs_data = cfg.pretransform_pipeline(pairs_data)
    train_pairs_data = cfg.data_train_pipeline(transformed_pairs_data, is_train=True)

    val_pairs_data = cfg.data_val_pipeline(transformed_pairs_data)
    test_pairs_data = cfg.data_test_pipeline(transformed_pairs_data)
    return {'train': train_pairs_data, 'val': val_pairs_data, 'test': test_pairs_data}


if __name__ == '__main__':
    config = create_config()
    data = get_api_data(config)




