import torch
import re
import collections
from torch._six import string_classes
import pandas as pd

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    def default_convert(data):
        r"""
            Function that converts each NumPy array element into a :class:`torch.Tensor`. If the input is a `Sequence`,
            `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
            If the input is not an NumPy array, it is left unchanged.
            This is used as the default function for collation when both `batch_sampler` and
            `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.
            The general input type to output type mapping is similar to that
            of :func:`~torch.utils.data.default_collate`. See the description there for more details.
            Args:
                data: a single data point to be converted
        """
        elem_type = type(data)
        if isinstance(data, torch.Tensor):
            return data
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            # array of string classes and object
            if elem_type.__name__ == 'ndarray' \
                    and np_str_obj_array_pattern.search(data.dtype.str) is not None:
                return data
            return torch.as_tensor(data)
        elif isinstance(data, collections.abc.Mapping):
            try:
                return elem_type({key: default_convert(data[key]) for key in data})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: default_convert(data[key]) for key in data}
        elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
            return elem_type(*(default_convert(d) for d in data))
        elif isinstance(data, tuple):
            return [default_convert(d) for d in data]  # Backwards compatibility.
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
            try:
                return elem_type([default_convert(d) for d in data])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_convert(d) for d in data]
        else:
            return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)  # , device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]
    elif isinstance(elem, pd.Index):  # is index
        return batch
    elif elem is None:  # index is not set
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))
