import pandas as pd
import tqdm
import torch
import logging
from torch.utils.data import Dataset
from typing import Dict, Generator, List

from epta.core.tool import Tool
from tilts.pipelines.block_augs import BlockAugs
from tilts.datasets.block_builders.block_builder import BlockBuilder


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, block_builder: 'BlockBuilder', inline: bool = True,
                 aug_pipeline: Dict[str, 'Tool'] = None):
        """
        Torch-like dataset from dataframe

        Args:
            df: dataframe to take data from
            block_builder: how to build blocks for dataset
            inline: True if to create block on call, not to cache
            aug_pipeline: called every __getitem__
        """
        logging.debug(f'Creating dataset with\n{block_builder}')
        self.df = df if df is not None else pd.DataFrame()
        self.block_builder = block_builder if block_builder is not None else BlockBuilder([])
        self.aug_pipeline = aug_pipeline if aug_pipeline is not None else BlockAugs([])

        # TODO: add from-API styled building
        self._inline = inline
        self.valid_block_indices = self._build_block_indices()
        self._data = self.__cache_dataset()

    def __create_elements(self) -> 'Generator':
        """
        :return: Creates set of indices where it is possible to build X and target blocks.
        """
        # print('Creating possible indexes for dataset and fitting criterions')
        logging.debug('Creating possible indexes for dataset and fitting criterions')
        for block_idx, _ in tqdm.tqdm(enumerate(self.df.index)):
            yield self._build_element(block_idx, mode='fit')

    def _build_block_indices(self) -> list:
        valid_indices = list()
        elements = list(self.__create_elements())
        for block_idx, element in enumerate(elements):
            if element is not None:
                valid_indices.append(block_idx)
        return valid_indices

    def __cache_dataset(self) -> list:
        data = list()
        # print(f'Caching data itself: {not self._inline}')
        logging.debug(f'Caching data itself: {not self._inline}')
        if not self._inline:  # if inline - create block in __getitem__ on call
            for _, target_idx in tqdm.tqdm(enumerate(self.valid_block_indices)):
                elem = self._build_element(target_idx, mode='transform')
                if elem is not None:
                    data.append(elem)
                else:
                    logging.warning('WARNING, some elems became None in fit/transform criterions')
        return data

    def _build_element(self, block_idx: int, mode: str = None):
        """
        :param block_idx: current position of block in df index from where to build it.
        :return: builds X and target blocks.
        """
        x = self.block_builder(self.df, block_idx, mode=mode)
        return x

    def _apply_block_augs(self, x: 'Block', **kwargs) -> 'Block':
        x = self.aug_pipeline(x, **kwargs)
        return x

    def _get_element(self, idx: int, **kwargs):
        if self._inline:
            # block_idx == index in global dataframe
            # idx == index of sample in dataset
            block_idx = self.valid_block_indices[idx]
            x = self._build_element(block_idx, mode='transform')
        else:
            # if dataset is already created
            x = self._data[idx]

        x = self._apply_block_augs(x, **kwargs.get('aug_kwargs', {}))

        elem = {
            'data': x.as_np(),
            'data_index': x.index,
            'data_columns': x.columns,
            'block_idx': x.idx,
            'dataset_element_idx': idx
        }
        # As not to re:implement collate_fn for batch sampling.
        # Adding idx for easier index getting.
        return elem

    def __getitem__(self, idx, **kwargs):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elem = self._get_element(idx, **kwargs)
        return elem

    def __len__(self):
        return len(self.valid_block_indices)


class MultipleTimeSeriesDataset(Dataset):
    def __init__(self, datasets: List[TimeSeriesDataset]):
        self.datasets = datasets
        # WARNING! Make sense only concatenating datasets made from same dfs.
        self.valid_indices = set(datasets[0].valid_block_indices)
        for dataset in datasets[1:]:
            self.valid_indices = self.valid_indices.intersection(set(dataset.valid_block_indices))
        self.valid_indices_sorted = sorted(list(self.valid_indices))
        self.block_idx_to_dataset_idx_mapping = list()
        for dataset in datasets:
            d = dict(zip(dataset.valid_block_indices, range(len(dataset.valid_block_indices))))
            self.block_idx_to_dataset_idx_mapping.append(d)

    def __getitem__(self, idx, **kwargs):
        result = list()
        selected_index = self.valid_indices_sorted[idx]
        for dataset, block_idx_to_dataset_idx_mapping in zip(self.datasets, self.block_idx_to_dataset_idx_mapping):
            dataset_idx = block_idx_to_dataset_idx_mapping[selected_index]
            result.append(dataset[dataset_idx])
        return result

    def __len__(self):
        return len(self.valid_indices)


if __name__ == '__main__':
    pass
