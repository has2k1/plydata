"""
Verb implementations for a :class:`pandas.DataFrame`
"""

import re
import warnings
from contextlib import suppress

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from .grouped_datatypes import GroupedDataFrame
from .options import get_option, options
from .utils import temporary_key, Q, get_empty_env, regular_index


def define(verb):
    if not get_option('modify_input_data'):
        verb.data = verb.data.copy()

    verb.env = verb.env.with_outer_namespace(_outer_namespace)
    with regular_index(verb.data):
        new_data = _evaluate_per_group(verb)
        for col in new_data:
            verb.data[col] = new_data[col]
    return verb.data


def create(verb):
    data = _get_base_dataframe(verb.data)
    verb.env = verb.env.with_outer_namespace(_outer_namespace)
    with regular_index(verb.data, data):
        new_data = _evaluate_per_group(verb)
        for col in new_data:
            data[col] = new_data[col]
    return data


def sample_n(verb):
    return verb.data.sample(**verb.kwargs)


def sample_frac(verb):
    return verb.data.sample(**verb.kwargs)


def select(verb):
    columns = verb.data.columns
    contains = verb.contains
    matches = verb.matches
    groups = _get_groups(verb)
    conds = []

    if verb.args:
        _args = set(verb.args)
        c1 = [x in _args for x in columns]
        conds.append(c1)

    if verb.startswith:
        c2 = [isinstance(x, str) and x.startswith(verb.startswith)
              for x in columns]
        conds.append(c2)

    if verb.endswith:
        c3 = [isinstance(x, str) and x.endswith(verb.endswith)
              for x in columns]
        conds.append(c3)

    if contains:
        c4 = []
        for col in columns:
            if isinstance(col, str):
                c4.append(any(s in col for s in contains))
            else:
                c4.append(False)
        conds.append(c4)

    if matches:
        c5 = []
        patterns = [x if hasattr(x, 'match') else re.compile(x)
                    for x in matches]
        for col in columns:
            if isinstance(col, str):
                c5.append(any(bool(p.match(col)) for p in patterns))
            else:
                c5.append(False)

        conds.append(c5)

    if groups:
        _groups = set(groups)
        c6 = [x in _groups for x in columns]
        conds.append(c6)

    if conds:
        cond = np.logical_or.reduce(conds)
    else:
        cond = np.array([False]*len(columns))

    if verb.drop:
        cond = ~cond

    data = verb.data.loc[:, cond]
    if data.is_copy:
        data.is_copy = None

    return data


def rename(verb):
    inplace = get_option('modify_input_data')
    data = verb.data.rename(columns=verb.lookup, inplace=inplace)
    return verb.data if inplace else data


def distinct(verb):
    if verb.new_columns:
        data = define(verb)
    else:
        data = verb.data
    return data.drop_duplicates(subset=verb.columns,
                                keep=verb.keep)


def arrange(verb):
    name_gen = ('col_{}'.format(x) for x in range(100))
    df = pd.DataFrame(index=verb.data.index)
    env = verb.env.with_outer_namespace({'Q': Q})
    for col, expr in zip(name_gen, verb.expressions):
        try:
            df[col] = verb.data[expr]
        except KeyError:
            df[col] = env.eval(expr, inner_namespace=verb.data)

    if len(df.columns):
        sorted_index = df.sort_values(by=list(df.columns)).index
        data = verb.data.loc[sorted_index, :]
    else:
        data = verb.data

    return data


def group_by(verb):
    if not all(g in verb.data for g in verb.groups):
        verb.data = define(verb)

    copy = not get_option('modify_input_data')
    return GroupedDataFrame(verb.data, verb.groups, copy=copy)


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
        data = _evaluate_per_group(verb, keep_index=False)
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
    if verb.single_function:
        verb.new_columns = None
        verb.expressions = verb.single_function
        keep_index = True
    else:
        verb.new_columns = verb.columns
        verb.expressions = verb.functions
        keep_index = False

    with regular_index(verb.data):
        data = _evaluate_per_group(verb, keep_index)

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


def tally(verb):
    # Prepare for summarize
    verb.new_columns = ['n']
    if verb.weights is not None:
        if isinstance(verb.weights, str):
            # Summarize will evaluate and sum up the weights
            verb.weights = 'sum({})'.format(verb.weights)
            verb.expressions = [verb.weights]
        else:
            # Do the summation here. The result does not depend
            # on the dataframe.
            verb.expressions = [np.sum(verb.weights)]
    else:
        verb.expressions = ['n()']

    data = summarize(verb)
    if verb.sort:
        data = data.sort_values(by='n', ascending=False)
        data.reset_index(drop=True, inplace=True)

    return data


def count(verb):
    if (not isinstance(verb.data, GroupedDataFrame) and
            verb.groups):
        verb.data = GroupedDataFrame(verb.data, verb.groups, copy=True)
    return tally(verb)


def add_tally(verb):
    verb.new_columns = ['n']

    if verb.weights:
        if isinstance(verb.weights, str):
            verb.weights = 'sum({})'.format(verb.weights)
        verb.expressions = [verb.weights]
    else:
        verb.expressions = ['n()']

    data = define(verb)
    if verb.sort:
        data = data.sort_values(by='n')
        data.reset_index(drop=True, inplace=True)

    return data


def add_count(verb):
    remove_groups = False
    if (not isinstance(verb.data, GroupedDataFrame) and
            verb.groups):
        verb.data = GroupedDataFrame(verb.data, verb.groups, copy=True)
        remove_groups = True

    data = add_tally(verb)
    if remove_groups:
        data = pd.DataFrame(data)
    return data


def modify_where(verb):
    if get_option('modify_input_data'):
        data = verb.data
    else:
        data = verb.data.copy()
    idx = data.query(verb.where, global_dict=verb.env.namespace).index
    qdf = data.loc[idx, :]

    env = verb.env.with_outer_namespace({'Q': Q})
    for col, expr in zip(verb.columns, verb.expressions):
        # Do not create new columns, define does that
        if col not in data:
            raise KeyError("Column '{}' not in dataframe".format(col))
        if isinstance(expr, str):
            data.loc[idx, col] = env.eval(expr, inner_namespace=qdf)
        else:
            data.loc[idx, col] = expr

    return data


def define_where(verb):
    if not get_option('modify_input_data'):
        verb.data = verb.data.copy()

    alt_expressions = [x[0] for x in verb.expressions]
    default_expressions = [x[1] for x in verb.expressions]

    with options(modify_input_data=True):
        verb.expressions = default_expressions
        verb.data = define(verb)

        verb.columns = verb.new_columns
        verb.expressions = alt_expressions
        data = modify_where(verb)

    return data


def dropna(verb):
    result = verb.data.dropna(
        axis=verb.axis,
        how=verb.how,
        thresh=verb.thresh,
        subset=verb.subset
    )
    return result


def fillna(verb):
    inplace = get_option('modify_input_data')
    result = verb.data.fillna(
        value=verb.value,
        method=verb.method,
        axis=verb.axis,
        limit=verb.limit,
        downcast=verb.downcast,
        inplace=inplace
    )
    return result if not inplace else verb.data


def call(verb):
    if isinstance(verb.func, str):
        func = getattr(verb.data, verb.func.lstrip('.'))
        return func(*verb.args, **verb.kwargs)
    else:
        return verb.func(verb.data, *verb.args, **verb.kwargs)


def inner_join(verb):
    verb.kwargs['how'] = 'inner'
    return _join(verb)


def outer_join(verb):
    verb.kwargs['how'] = 'outer'
    return _join(verb)


def left_join(verb):
    verb.kwargs['how'] = 'left'
    return _join(verb)


def right_join(verb):
    verb.kwargs['how'] = 'right'
    return _join(verb)


def anti_join(verb):
    verb.kwargs['how'] = 'left'
    verb.kwargs['suffixes'] = ('', '_y')
    verb.kwargs['indicator'] = '_plydata_merge'
    df = _join(verb)
    data = df.query('_plydata_merge=="left_only"')[verb.x.columns]
    data.is_copy = None
    return data


def semi_join(verb):
    verb.kwargs['how'] = 'left'
    verb.kwargs['suffixes'] = ('', '_y')
    verb.kwargs['indicator'] = '_plydata_merge'
    df = _join(verb)
    data = df.query('_plydata_merge=="both"')[verb.x.columns]
    data.is_copy = None
    data.drop_duplicates(inplace=True)
    return data


# Helper functions

def _get_groups(verb):
    """
    Return groups
    """
    try:
        return verb.data.plydata_groups
    except AttributeError:
        return []


def _get_base_dataframe(df):
    """
    Remove all columns other than those grouped on
    """
    if isinstance(df, GroupedDataFrame):
        base_df = GroupedDataFrame(
            df.loc[:, df.plydata_groups], df.plydata_groups,
            copy=True)
    else:
        base_df = pd.DataFrame(index=df.index)
    return base_df


def _add_group_columns(data, gdf):
    """
    Add group columns to data with a value from the grouped dataframe

    It is assumed that the grouped dataframe contains a single group

    >>> data = pd.DataFrame({
    ...     'x': [5, 6, 7]})
    >>> gdf = GroupedDataFrame({
    ...     'g': list('aaa'),
    ...     'x': range(3)}, groups=['g'])
    >>> _add_group_columns(data, gdf)
       g  x
    0  a  5
    1  a  6
    2  a  7
    """
    n = len(data)
    if isinstance(gdf, GroupedDataFrame):
        for i, col in enumerate(gdf.plydata_groups):
            if col not in data:
                group_values = [gdf[col].iloc[0]] * n
                # Need to be careful and maintain the dtypes
                # of the group columns
                if pdtypes.is_categorical_dtype(gdf[col]):
                    col_values = pd.Categorical(
                        group_values,
                        categories=gdf[col].cat.categories,
                        ordered=gdf[col].cat.ordered
                    )
                else:
                    col_values = pd.Series(
                        group_values,
                        index=data.index,
                        dtype=gdf[col].dtype
                    )
                # Group columns come first
                data.insert(i, col, col_values)
    return data


def _create_column(data, col, value):
    """
    Create column in dataframe

    Helper method meant to deal with problematic
    column values. e.g When the series index does
    not match that of the data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe in which to insert value
    col : column label
        Column name
    value : object
        Value to assign to column

    Returns
    -------
    data : pandas.DataFrame
        Modified original dataframe

    >>> df = pd.DataFrame({'x': [1, 2, 3]})
    >>> y = pd.Series([11, 12, 13], index=[21, 22, 23])

    Data index and value index do not match

    >>> _create_column(df, 'y', y)
       x   y
    0  1  11
    1  2  12
    2  3  13

    Non-empty dataframe, scalar value

    >>> _create_column(df, 'z', 3)
       x   y  z
    0  1  11  3
    1  2  12  3
    2  3  13  3

    Empty dataframe, scalar value

    >>> df = pd.DataFrame()
    >>> _create_column(df, 'w', 3)
       w
    0  3
    >>> _create_column(df, 'z', 'abc')
       w    z
    0  3  abc
    """
    with suppress(AttributeError):
        # If the index of a series and the dataframe
        # in which the series will be assigned to a
        # column do not match, missing values/NaNs
        # are created. We do not want that.
        if not value.index.equals(data.index):
            if len(value) == len(data):
                value.index = data.index
            else:
                value.reset_index(drop=True, inplace=True)

    # You cannot assign a scalar value to a dataframe
    # without an index. You need an interable value.
    if data.index.empty:
        try:
            len(value)
        except TypeError:
            scalar = True
        else:
            scalar = isinstance(value, str)

        if scalar:
            value = [value]

    data[col] = value
    return data


def _evaluate_per_group(verb, keep_index=True):
    """
    Evaluate Expressions and return the columns in a new dataframe

    Parameters
    ----------
    verb : object
        verb.data should have a regular index (RangeIndex).
    keep_index : bool
        If True, the evaluation method *tries* to create a
        dataframe with the same index as the original.
        Ultimately whether this succeeds depends on the
        expressions to be evaluated.
    """
    # assert isinstance(verb.data.index, pd.RangeIndex)
    try:
        groups = verb.data.plydata_groups
    except AttributeError:
        data = _evaluate(verb, verb.data, keep_index)
    else:
        grouper = verb.data.groupby(groups, sort=False)
        dfs = [_evaluate(verb, gdf, keep_index) for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=False, copy=False)

        # groupby can mixup the rows. We try to maintain the original
        # order, but we can only do that if the result has a one to
        # one relationship with the original
        one2one = (
            keep_index and
            not any(data.index.duplicated()) and
            len(data.index) == len(verb.data.index))
        if one2one:
            data = data.sort_index()
        else:
            data.reset_index(drop=True, inplace=True)

        # Maybe this should happen in the verb functions
        try:
            keep_groups = verb.keep_groups
        except AttributeError:
            keep_groups = True
        finally:
            if keep_groups:
                data.plydata_groups = list(groups)

    return data


def _evaluate(verb, gdf, keep_index=True):
    """
    Evaluate verb expressions and return a new dataframe

    Parameters
    ----------
    verb : object
        An object with *expressions*, *new_columns* and
        *env* attributes.
    gdf : dataframe
        Dataframe where all items are assumed to belong to the
        same group.
    keep_index : bool
        If True, the evaluation method *tries* to create a
        dataframe with the same index as the original.
        Ultimately whether this succeeds depends on the
        expressions to be evaluated.

    This excutes an *apply* part of the *split-apply-combine*
    data manipulation strategy. Callers of the function do
    the *split* and the *combine*.

    A peak into the function

    >>> import pandas as pd
    >>> from .utils import get_empty_env

    A Dataframe with many groups

    >>> df = GroupedDataFrame(
    ...     {'x': list('aabbcc'),
    ...      'y': [0, 1, 2, 3, 4, 5]
    ...      }, groups=['x'])

    Do a groupby and obtain a dataframe with a single group,
    (i.e. the *split*)

    >>> grouper = df.groupby(df.plydata_groups)
    >>> group = 'b'      # The group we want to evalualte
    >>> gdf = grouper.get_group(group)
    >>> gdf
    groups: ['x']
      x  y
    2 b  2
    3 b  3

    Create the other parameters
    >>> verb = type('verb', (object,), {})
    >>> verb.env = get_empty_env()
    >>> verb.new_columns = ['ysq', 'ycubed']
    >>> verb.expressions = ['y**2', 'y**3']

    Finally, the *apply*

    >>> _evaluate(verb, gdf)
       x  ysq  ycubed
    2  b    4       8
    3  b    9      27

    The caller does the *combine*, for one or more of these
    results.
    """
    def n():
        """
        Return number of rows in groups

        This function is part of the public API
        """
        return len(gdf)

    gdf.is_copy = None
    result_index = gdf.index if keep_index else []
    data = pd.DataFrame(index=result_index)
    expressions = verb.expressions
    new_columns = verb.new_columns

    if expressions is not None and new_columns is not None:
        for col, expr in zip(new_columns, expressions):
            if isinstance(expr, str):
                d = dict(gdf, n=n)
                value = verb.env.eval(expr, inner_namespace=d)
            elif callable(expr):
                value = expr(gdf)
            else:
                value = expr

            _create_column(data, col, value)
    elif callable(expressions):
        data = expressions(gdf)
    else:
        raise TypeError(
            "{} cannot evaluate object of type "
            "{}".format(type(verb), type(verb.expressions)))

    data = _add_group_columns(data, gdf)
    return data


def _join(verb):
    """
    Join helper
    """
    data = pd.merge(verb.x, verb.y, **verb.kwargs)

    # Preserve x groups
    if isinstance(verb.x, GroupedDataFrame):
        data.plydata_groups = list(verb.x.plydata_groups)
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
