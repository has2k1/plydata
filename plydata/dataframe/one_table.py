"""
One table verb implementations for a :class:`pandas.DataFrame`
"""
import warnings

import numpy as np
import pandas as pd

from ..types import GroupedDataFrame
from ..options import get_option
from ..utils import Q, get_empty_env, regular_index, unique
from .common import Evaluator, Selector
from .common import _get_groups, _get_base_dataframe

__all__ = ['arrange', 'create', 'define', 'distinct', 'do',
           'group_by', 'group_indices', 'head',  'mutate',
           'pull', 'query', 'rename', 'sample_frac', 'sample_n',
           'select', 'slice_rows', 'summarize', 'tail',
           'ungroup', 'unique']


def define(verb):
    if not get_option('modify_input_data'):
        verb.data = verb.data.copy()

    if not verb.expressions:
        return verb.data

    verb.env = verb.env.with_outer_namespace(_outer_namespace)
    with regular_index(verb.data):
        new_data = Evaluator(verb).process()
        for col in new_data:
            verb.data[col] = new_data[col]
    return verb.data


def create(verb):
    data = _get_base_dataframe(verb.data)
    verb.env = verb.env.with_outer_namespace(_outer_namespace)
    with regular_index(verb.data, data):
        new_data = Evaluator(verb, drop=True).process()
        for col in new_data:
            data[col] = new_data[col]
    return data


def sample_n(verb):
    return verb.data.sample(**verb.kwargs)


def sample_frac(verb):
    return verb.data.sample(**verb.kwargs)


def select(verb):
    columns = Selector.get(verb)
    data = verb.data.loc[:, columns]
    return data


def rename(verb):
    inplace = get_option('modify_input_data')
    data = verb.data.rename(columns=verb.lookup, inplace=inplace)
    return verb.data if inplace else data


def distinct(verb):
    data = define(verb)
    return data.drop_duplicates(subset=verb.columns,
                                keep=verb.keep)


def arrange(verb):
    # Do not evaluate if all statements correspond to
    # columns already in the dataframe
    stmts = [expr.stmt for expr in verb.expressions]
    has_all_columns = all(stmt in verb.data for stmt in stmts)
    if has_all_columns:
        df = verb.data.loc[:, stmts]
    else:
        verb.env = verb.env.with_outer_namespace({'Q': Q})
        df = Evaluator(verb, keep_index=True).process()

    if len(df.columns):
        sorted_index = df.sort_values(by=list(df.columns)).index
        data = verb.data.loc[sorted_index, :]
    else:
        data = verb.data
    return data


def group_by(verb):
    verb.data = define(verb)

    copy = not get_option('modify_input_data')

    try:
        add = verb.add_
    except AttributeError:
        add = False

    if add:
        groups = _get_groups(verb) + verb.groups
    else:
        groups = verb.groups

    if groups:
        return GroupedDataFrame(verb.data, groups, copy=copy)
    else:
        return pd.DataFrame(verb.data, copy=copy)


def ungroup(verb):
    return pd.DataFrame(verb.data)


def group_indices(verb):
    data = verb.data
    groups = verb.groups
    if isinstance(data, GroupedDataFrame):
        if groups:
            msg = "GroupedDataFrame ignored extra groups {}"
            warnings.warn(msg.format(groups))
        else:
            groups = data.plydata_groups
    else:
        data = create(verb)

    indices_dict = data.groupby(groups, sort=False).indices
    indices = -np.ones(len(data), dtype=int)
    for i, (_, idx) in enumerate(sorted(indices_dict.items())):
        indices[idx] = i

    return indices


def summarize(verb):
    verb.env = verb.env.with_outer_namespace(_outer_namespace)
    with regular_index(verb.data):
        data = Evaluator(
            verb,
            keep_index=False,
            keep_groups=False).process()
    return data


def query(verb):
    data = verb.data.query(
        verb.expression,
        global_dict=verb.env.namespace,
        **verb.kwargs)
    data.is_copy = None
    return data


def do(verb):
    verb.env = get_empty_env()
    keep_index = verb.single_function
    if verb.single_function:
        if isinstance(verb.expressions[0].stmt, str):
            raise TypeError(
                "A single function for `do` cannot be a string")

    with regular_index(verb.data):
        data = Evaluator(verb, keep_index=keep_index).process()

    if (len(verb.data.index) == len(data.index)):
        data.index = verb.data.index

    return data


def head(verb):
    if isinstance(verb.data, GroupedDataFrame):
        grouper = verb.data.groupby(verb.data.plydata_groups, sort=False)
        dfs = [gdf.head(verb.n) for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=True, copy=False)
        data.plydata_groups = list(verb.data.plydata_groups)
    else:
        data = verb.data.head(verb.n)

    return data


def tail(verb):
    if isinstance(verb.data, GroupedDataFrame):
        grouper = verb.data.groupby(verb.data.plydata_groups, sort=False)
        dfs = [gdf.tail(verb.n) for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=True, copy=False)
        data.plydata_groups = list(verb.data.plydata_groups)
    else:
        data = verb.data.tail(verb.n)

    return data


def pull(verb):
    copy = not get_option('modify_input_data')

    if verb.use_index:
        s = verb.data.iloc[:, verb.column]
    else:
        s = verb.data.loc[:, verb.column]

    if copy:
        s = s.copy()

    return np.asarray(s)


def slice_rows(verb):
    copy = not get_option('modify_input_data')
    data = verb.data.iloc[verb.slice, :]
    if copy:
        data = data.copy()
    return data


# Aggregations functions

def _nth(arr, n):
    """
    Return the nth value of array

    If it is missing return NaN
    """
    try:
        return arr.iloc[n]
    except (KeyError, IndexError):
        return np.nan


def _n_distinct(arr):
    """
    Number of unique values in array
    """
    return len(pd.unique(arr))


_outer_namespace = {
    'min': np.min,
    'max': np.max,
    'sum': np.sum,
    'cumsum': np.cumsum,
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'first': lambda x: _nth(x, 0),
    'last': lambda x: _nth(x, -1),
    'nth': _nth,
    'n_distinct': _n_distinct,
    'n_unique': _n_distinct,
    'Q': Q
}

# Aliases
mutate = define
