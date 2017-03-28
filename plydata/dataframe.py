"""
Verb implementations for a :class:`pandas.DataFrame`
"""

import re

import numpy as np
import pandas as pd

from .grouped_datatypes import GroupedDataFrame
from .utils import hasattrs


def mutate(verb):
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            value = verb.env.eval(expr, inner_namespace=verb.data)
        elif hasattr(expr, '__len__'):
            if len(verb.data) == len(expr):
                value = expr
            else:
                msg = "value not equal to length of dataframe"
                raise ValueError(msg)
        else:
            msg = "Cannot handle expression of type `{}`"
            raise TypeError(msg.format(type(expr)))
        verb.data[col] = value
    return verb.data


def transmute(verb):
    data = _get_base_dataframe(verb.data)
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            data[col] = verb.env.eval(expr, inner_namespace=verb.data)
        elif hasattr(expr, '__len__'):
            if len(verb.data) == len(expr):
                data[col] = expr
            else:
                msg = "value not equal to length of dataframe"
                raise ValueError(msg)
        else:
            msg = "Cannot handle expression of type `{}`"
            raise TypeError(msg.format(type(expr)))

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
        data = data.copy()

    return data


def rename(verb):
    return verb.data.rename(columns=verb.lookup)


def distinct(verb):
    if hasattrs(verb, ('new_columns', 'expressions')):
        mutate(verb)
    return verb.data.drop_duplicates(subset=verb.columns,
                                     keep=verb.keep)


def arrange(verb):
    name_gen = ('col_{}'.format(x) for x in range(100))
    columns = []
    d = {}
    for col, expr in zip(name_gen, verb.expressions):
        d[col] = verb.env.eval(expr, inner_namespace=verb.data)
        columns.append(col)

    if columns:
        df = pd.DataFrame(d).sort_values(by=columns)
        data = verb.data.loc[df.index, :]
        if data.is_copy:
            data = data.copy()
    else:
        data = verb.data

    return data


def group_by(verb):
    data = GroupedDataFrame(verb.data)
    data.plydata_groups = verb.groups
    mutate(verb)
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
        base_df = GroupedDataFrame(df.loc[:, df.plydata_groups])
        base_df.plydata_groups = list(df.plydata_groups)
    else:
        base_df = pd.DataFrame(index=df.index)
    return base_df
