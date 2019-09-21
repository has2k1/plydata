"""
Tidy verb initializations
"""
from itertools import chain
from warnings import warn

import numpy as np
import pandas as pd

from .common import Selector
from ..utils import convert_str, identity, clean_indices


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


def separate(verb):
    data = verb.data
    col = data[verb.col]
    npieces = len(verb.into)
    nsplits = npieces - 1
    exclude_columns = pd.isnull(verb.into)
    pattern = verb._pattern
    positions = verb._positions

    # bookkeeping for extra or fewer pieces than npieces
    warn_extra = verb.extra == 'warn'
    warn_fewer = verb.fill == 'warn'
    extra_rows = []
    fewer_rows = []
    fill_side = 'right' if warn_fewer else verb.fill

    if verb.extra == 'merge':
        maxsplit = nsplits
    else:
        maxsplit = 0

    def split_at_pattern(s, i):
        """Split and note any indices for the warnings"""
        if pd.isnull(s):
            return [s] * npieces

        res = pattern.split(s, maxsplit=maxsplit)
        diff = npieces - len(res)

        if diff < 0:
            if warn_extra:
                extra_rows.append(i)
            res = res[:npieces]
        elif diff > 0:
            if warn_fewer:
                fewer_rows.append(i)
            if fill_side == 'right':
                res += [None] * diff
            else:
                res = [None] * diff + res
        return res

    def split_at_positions(s):
        """Split"""
        return [s[i:j] for i, j in zip(positions[:-1], positions[1:])]

    if pattern:
        splits = [split_at_pattern(s, i) for i, s in enumerate(col)]
    else:
        splits = [split_at_positions(s) for s in col]

    split_df = pd.DataFrame(splits, columns=verb.into)
    if exclude_columns.any():
        split_df = split_df.loc[:, ~exclude_columns]

    if warn_extra and extra_rows:
        warn("Expected {} pieces: Additional pieces discarded "
             "in {} rows: {}".format(
                 npieces,
                 len(extra_rows),
                 extra_rows
             ))

    if warn_fewer and fewer_rows:
        warn("Expected {} pieces: Missing pieces filled with "
             "`None` in {} rows: {}".format(
                 npieces,
                 len(fewer_rows),
                 fewer_rows
             ))

    if verb.convert:
        split_df = convert_str(split_df)

    # Insert the created columns
    col_location = data.columns.get_loc(verb.col)
    if verb.remove:
        stop, start = col_location, col_location+1
    else:
        stop, start = col_location+1, col_location+1

    data = pd.concat(
        [
            data.iloc[:, :stop],
            split_df,
            data.iloc[:, start:]
        ],
        axis=1,
        copy=False
    )
    return data


def pivot_wider(verb):
    if verb.id_cols:
        index = verb.id_cols
    else:
        index = verb.data.columns.difference(
            set(verb.names_from) | set(verb.values_from)
        ).tolist()

    # Any extra columns should not appear in the output
    exclude = verb.data.columns.difference(
        list(chain(index, verb.names_from, verb.values_from))
    )
    data = verb.data.drop(exclude, axis=1) if len(exclude) else verb.data
    data = pd.pivot_table(
        data,
        index=index,
        columns=verb.names_from,
        aggfunc=verb.values_fn,
        fill_value=verb.values_fill,
    )

    if verb.names_prefix:
        # Get innermost (last) level,
        # add prefix to it and create a new level
        # add the new (renamed) level to the previous levels
        levels = data.columns.levels
        last_level = levels[-1]
        renamed_last_level = pd.Index(
            (verb.names_prefix + str(x) for x in last_level),
            name=last_level.name
        )
        renamed_levels = tuple(levels[:-1]) + (renamed_last_level,)
        data.columns = data.columns.set_levels(renamed_levels)

    clean_indices(data, verb.names_sep, inplace=True)
    data = data.infer_objects()
    return data
