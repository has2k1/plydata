"""
Verb implementations for a :class:`pandas.DataFrame`
"""

import re
import warnings

import numpy as np
import pandas as pd

from .grouped_datatypes import GroupedDataFrame
from .options import get_option
from .utils import hasattrs, temporary_key


def define(verb):
    if get_option('modify_input_data'):
        data = verb.data
    else:
        data = verb.data.copy()

    new_data = _evaluate_expressions(verb)
    for col in new_data:
        data[col] = new_data[col]
    return data


def create(verb):
    data = _get_base_dataframe(verb.data)
    new_data = _evaluate_expressions(verb)
    for col in new_data:
        data[col] = new_data[col]
    return data


def sample_n(verb):
    return verb.data.sample(**verb.kwargs)


def sample_frac(verb):
    return verb.data.sample(**verb.kwargs)


def select(verb):
    kw = verb.kwargs
    columns = verb.data.columns
    groups = _get_groups(verb)
    c0 = np.array([False]*len(columns))
    c1 = c2 = c3 = c4 = c5 = c6 = c0

    if verb.args:
        c1 = [x in set(verb.args) for x in columns]

    if kw['startswith']:
        c2 = [isinstance(x, str) and x.startswith(kw['startswith'])
              for x in columns]

    if kw['endswith']:
        c3 = [isinstance(x, str) and x.endswith(kw['endswith'])
              for x in columns]

    if kw['contains']:
        c4 = [isinstance(x, str) and kw['contains'] in x
              for x in columns]

    if kw['matches']:
        if hasattr(kw['matches'], 'match'):
            pattern = kw['matches']
        else:
            pattern = re.compile(kw['matches'])
        c5 = [isinstance(x, str) and bool(pattern.match(x))
              for x in columns]

    if groups:
        c6 = [x in set(groups) for x in columns]

    cond = np.logical_or.reduce((c1, c2, c3, c4, c5, c6))

    if kw['drop']:
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
    if hasattrs(verb, ('new_columns', 'expressions')):
        data = define(verb)
    else:
        data = verb.data
    return data.drop_duplicates(subset=verb.columns,
                                keep=verb.keep)


def arrange(verb):
    name_gen = ('col_{}'.format(x) for x in range(100))
    df = pd.DataFrame(index=verb.data.index)
    for col, expr in zip(name_gen, verb.expressions):
        df[col] = verb.env.eval(expr, inner_namespace=verb.data)

    if len(df.columns):
        sorted_index = df.sort_values(by=list(df.columns)).index
        data = verb.data.loc[sorted_index, :]
        if data.is_copy:
            data.is_copy = None
    else:
        data = verb.data

    return data


def group_by(verb):
    copy = not get_option('modify_input_data')
    verb.data = GroupedDataFrame(verb.data, verb.groups, copy=copy)
    return define(verb)


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

    indices_dict = data.groupby(groups).indices
    indices = -np.ones(len(data), dtype=int)
    for i, (_, idx) in enumerate(sorted(indices_dict.items())):
        indices[idx] = i

    return indices


def summarize(verb):
    env = verb.env
    cols = verb.new_columns
    exprs = verb.expressions
    data = verb.data

    try:
        grouper = data.groupby(data.plydata_groups)
    except AttributeError:
        data = _eval_summarize_expressions(exprs, cols, env, data)
    else:
        dfs = [_eval_summarize_expressions(exprs, cols, env, gdf)
               for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=True)

    return data


def query(verb):
    data = verb.data.query(verb.expression, **verb.kwargs)
    if data.is_copy:
        data.is_copy = None
    return data


def do(verb):
    if verb.single_function:
        return _do_single_function(verb)
    else:
        return _do_functions(verb)


def head(verb):
    if isinstance(verb.data, GroupedDataFrame):
        grouper = verb.data.groupby(verb.data.plydata_groups)
        dfs = [gdf.head(verb.n) for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=True)
        data.plydata_groups = list(verb.data.plydata_groups)
    else:
        data = verb.data.head(verb.n)

    return data


def tail(verb):
    if isinstance(verb.data, GroupedDataFrame):
        grouper = verb.data.groupby(verb.data.plydata_groups)
        dfs = [gdf.tail(verb.n) for _, gdf in grouper]
        data = pd.concat(dfs, axis=0, ignore_index=True)
        data.plydata_groups = list(verb.data.plydata_groups)
    else:
        data = verb.data.tail(verb.n)

    return data


def tally(verb):
    # Prepare for summarize
    verb.new_columns = ['n']
    if verb.weights:
        if isinstance(verb.weights, str):
            verb.weights = 'sum({})'.format(verb.weights)
        verb.expressions = [verb.weights]
    else:
        verb.expressions = ['{n}']

    data = summarize(verb)
    if verb.sort:
        data = data.sort_values(by='n')
        data.reset_index(drop=True, inplace=True)

    return data


def count(verb):
    if (not isinstance(verb.data, GroupedDataFrame) and
            verb.groups):
        verb.data = GroupedDataFrame(verb.data, verb.groups, copy=True)
    return tally(verb)


def modify_where(verb):
    if get_option('modify_input_data'):
        data = verb.data
    else:
        data = verb.data.copy()
    idx = data.query(verb.where).index
    qdf = data.loc[idx, :]

    for col, expr in zip(verb.columns, verb.expressions):
        # Do not create new columns, define does that
        if col not in data:
            raise KeyError("Column '{}' not in dataframe".format(col))
        if isinstance(expr, str):
            data.loc[idx, col] = verb.env.eval(expr, inner_namespace=qdf)
        else:
            data.loc[idx, col] = expr

    return data


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


def _evaluate_expressions(verb):
    """
    Evaluate Expressions and return the columns in a new dataframe
    """
    data = pd.DataFrame(index=verb.data.index)
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            data[col] = verb.env.eval(expr, inner_namespace=verb.data)
        else:
            data[col] = expr

    return data


def _eval_summarize_expressions(expressions, columns, env, gdf):
    """
    Evaluate expressions to create a dataframe.

    Parameters
    ----------
    expressions : list of str
    columns : list of str
    env : environment
    gdf : dataframe
        Dataframe where all items are assumed to belong to the
        same group.

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

    >>> env = get_empty_env()
    >>> columns = ['ysq', 'ycubed']
    >>> expressions = ['y**2', 'y**3']

    Finally, the *apply*

    >>> _eval_summarize_expressions(expressions, columns, env, gdf)
       x  ysq  ycubed
    0  b    4       8
    1  b    9      27

    The caller does the *combine*, for one or more of these
    results.
    """
    env = env.with_outer_namespace(_aggregate_functions)

    # Extra aggregate function that the user references with
    # the name `{n}`. It returns the length of the dataframe.
    def _plydata_n():
        return len(gdf)

    for col, expr in zip(columns, expressions):
        if isinstance(expr, str):
            expr = expr.format(n='_plydata_n()')
            with temporary_key(_aggregate_functions,
                               '_plydata_n', _plydata_n):
                value = env.eval(expr, inner_namespace=gdf)
        else:
            value = expr

        # Non-consecutive 0-n pd.series indices create gaps with
        # nans when inserted into a dataframe
        value = np.asarray(value)

        try:
            data[col] = value
        except NameError:
            try:
                n = len(value)
            except TypeError:
                n = 1
            data = pd.DataFrame({col: value}, index=range(n))

    # Add the grouped-on columns
    if isinstance(gdf, GroupedDataFrame):
        for i, col in enumerate(gdf.plydata_groups):
            data.insert(i, col, gdf[col].iloc[0])

    return data


def _eval_do_single_function(function, gdf):
    """
    Evaluate an expression to create a dataframe.

    Similar to :func:`_eval_summarize_expressions`, but for the
    ``do`` operation.
    """
    gdf.is_copy = None
    data = function(gdf)

    # Add the grouped-on columns
    if isinstance(gdf, GroupedDataFrame):
        for i, col in enumerate(gdf.plydata_groups):
            if col not in data:
                data.insert(i, col, gdf[col].iloc[0])

    return data


def _eval_do_functions(functions, columns, gdf):
    """
    Evaluate functions to create a dataframe.

    Similar to :func:`_eval_summarize_expressions`, but for the
    ``do`` operation.
    """
    gdf.is_copy = None
    for col, func in zip(columns, functions):
        value = np.asarray(func(gdf))

        try:
            data[col] = value
        except NameError:
            try:
                n = len(value)
            except TypeError:
                n = 1
            data = pd.DataFrame({col: value}, index=range(n))

    # Add the grouped-on columns
    if isinstance(gdf, GroupedDataFrame):
        for i, col in enumerate(gdf.plydata_groups):
            if col not in data:
                data.insert(i, col, gdf[col].iloc[0])

    return data


def _do_single_function(verb):
    func = verb.single_function
    data = verb.data

    try:
        groups = data.plydata_groups
    except AttributeError:
        data = _eval_do_single_function(func, data)
    else:
        dfs = [_eval_do_single_function(func, gdf)
               for _, gdf in data.groupby(groups)]
        data = pd.concat(dfs, axis=0, ignore_index=True)
        data.plydata_groups = list(groups)

    return data


def _do_functions(verb):
    cols = verb.columns
    funcs = verb.functions
    data = verb.data

    try:
        groups = data.plydata_groups
    except AttributeError:
        data = _eval_do_functions(funcs, cols, data)
    else:
        dfs = [_eval_do_functions(funcs, cols, gdf)
               for _, gdf in data.groupby(groups)]
        data = pd.concat(dfs, axis=0, ignore_index=True)
        data.plydata_groups = list(groups)

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
    except KeyError:
        raise np.nan


def _n_distinct(arr):
    """
    Number of unique values in array
    """
    return len(np.unique(arr))


_aggregate_functions = {
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
}
