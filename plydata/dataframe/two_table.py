"""
Two table verb implementations for a :class:`pandas.DataFrame`
"""
import pandas as pd

from ..types import GroupedDataFrame
from ..operators import register_implementations

__all__ = [
    'inner_join', 'outer_join', 'left_join',
    'right_join', 'anti_join', 'semi_join'
]


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
    data._is_copy = None
    return data


def semi_join(verb):
    verb.kwargs['how'] = 'left'
    verb.kwargs['suffixes'] = ('', '_y')
    verb.kwargs['indicator'] = '_plydata_merge'
    df = _join(verb)
    data = df.query('_plydata_merge=="both"')[verb.x.columns]
    data._is_copy = None
    data.drop_duplicates(inplace=True)
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


register_implementations(globals(), __all__, 'dataframe')
