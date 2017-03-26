"""
Verb implementations for a :class:`pandas.DataFrame`
"""

import re

import numpy as np
import pandas as pd

from .utils import hasattrs


def mutate(verb):
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            value = verb.env.eval(expr, inner_namespace=verb.data)
        elif len(verb.data) == len(value):
            value = expr
        else:
            raise ValueError("Unknown type")
        verb.data[col] = value
    return verb.data


def transmute(verb):
    d = {}
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            value = verb.env.eval(expr, inner_namespace=verb.data)
        elif len(verb.data) == len(value):
            value = expr
        else:
            raise ValueError("Unknown type")
        d[col] = value

    if d:
        data = pd.DataFrame(d)
    else:
        data = pd.DataFrame(index=verb.data.index)

    return data


def sample_n(verb):
    return verb.data.sample(**verb.kwargs)


def sample_frac(verb):
    return verb.data.sample(**verb.kwargs)


def select(verb):
    kw = verb.kwargs
    columns = verb.data.columns
    c0 = np.array([False]*len(columns))
    c1 = c2 = c3 = c4 = c5 = c0

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

    cond = np.asarray(c1) | c2 | c3 | c4 | c5

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
    data = verb.data
    name_gen = ('col_{}'.format(x) for x in range(100))
    columns = []
    d = {}
    for col, expr in zip(name_gen, verb.expressions):
        d[col] = verb.env.eval(expr, inner_namespace=verb.data)
        columns.append(col)

    if columns:
        df = pd.DataFrame(d).sort_values(by=columns)
        data = data.loc[df.index, :]
        if data.is_copy:
            data = data.copy()

    return data
