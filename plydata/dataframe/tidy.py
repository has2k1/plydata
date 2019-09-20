"""
Tidy verb initializations
"""
from itertools import chain
from warnings import warn

import numpy as np
import pandas as pd

from .common import Selector
from ..utils import identity, clean_indices


def gather(verb):
    data = verb.data
    verb._select_verb.data = data
    columns = Selector.get(verb._select_verb)
    exclude = pd.Index(columns).drop_duplicates()
    id_vars = data.columns.difference(exclude, sort=False)
    return pd.melt(data, id_vars, columns, verb.key, verb.value)


def spread(verb):
    key = verb.key
    value = verb.value

    if isinstance(key, str) or not np.iterable(key):
        key = [key]

    if isinstance(value, str) or not np.iterable(key):
        value = [value]

    key_value = pd.Index(list(chain(key, value))).drop_duplicates()
    index = verb.data.columns.difference(key_value).tolist()
    data = pd.pivot_table(
        verb.data,
        values=value,
        index=index,
        columns=key,
        aggfunc=identity,
    )

    clean_indices(data, verb.sep, inplace=True)
    data = data.infer_objects()
    return data
