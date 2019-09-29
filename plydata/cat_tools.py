"""
Functions for categoricals
"""

import pandas as pd
import pandas.api.types as pdtypes
from pandas.core.algorithms import value_counts

__all__ = [
    'cat_infreq',
    'cat_inorder',
]


def cat_infreq(c, ordered=None):
    """
    Reorder categorical by frequency of the values

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    ordered : bool
        If ``True``, the categorical is ordered.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> x = ['d', 'a', 'b', 'b', 'c', 'c', 'c']
    >>> cat_infreq(x)
    [d, a, b, b, c, c, c]
    Categories (4, object): [c, b, d, a]
    >>> cat_infreq(x, ordered=True)
    [d, a, b, b, c, c, c]
    Categories (4, object): [c < b < d < a]

    When two or more values occur the same number of times, if the
    categorical is ordered, the order is preserved. If it is not
    not ordered, the order depends on that of the values. Above 'd'
    comes before 'a', and below 'a' comes before 'a'.

    >>> c = pd.Categorical(
    ...     x, categories=['a', 'c', 'b', 'd']
    ... )
    >>> cat_infreq(c)
    [d, a, b, b, c, c, c]
    Categories (4, object): [c, b, a, d]
    >>> cat_infreq(c.set_ordered(True))
    [d, a, b, b, c, c, c]
    Categories (4, object): [c < b < a < d]
    """
    kwargs = {} if ordered is None else {'ordered': ordered}
    counts = value_counts(c)
    if pdtypes.is_categorical(c):
        original_cat_order = c.categories
    else:
        original_cat_order = pd.unique(c)
    counts = counts.reindex(index=original_cat_order)
    cats = (_stable_series_sort(counts, ascending=False)
            .index
            .to_list())
    return pd.Categorical(c, categories=cats, **kwargs)


def cat_inorder(c, ordered=None):
    """
    Reorder categorical by appearance

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    ordered : bool
        If ``True``, the categorical is ordered.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> import numpy as np
    >>> x = [4, 1, 3, 4, 4, 7, 3]
    >>> cat_inorder(x)
    [4, 1, 3, 4, 4, 7, 3]
    Categories (4, int64): [4, 1, 3, 7]
    >>> arr = np.array(x)
    >>> cat_inorder(arr)
    [4, 1, 3, 4, 4, 7, 3]
    Categories (4, int64): [4, 1, 3, 7]
    >>> c = ['b', 'f', 'c', None, 'c', 'a', 'b', 'e']
    >>> cat_inorder(c)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b, f, c, a, e]
    >>> s = pd.Series(c)
    >>> cat_inorder(s)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b, f, c, a, e]
    >>> cat = pd.Categorical(c)
    >>> cat_inorder(cat)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b, f, c, a, e]
    >>> cat_inorder(cat, ordered=True)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b < f < c < a < e]

    By default, ordered categories remain ordered.

    >>> ocat = pd.Categorical(cat, ordered=True)
    >>> ocat
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [a < b < c < e < f]
    >>> cat_inorder(ocat)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b < f < c < a < e]
    >>> cat_inorder(ocat, ordered=False)
    [b, f, c, NaN, c, a, b, e]
    Categories (5, object): [b, f, c, a, e]

    Notes
    -----
    ``NaN`` or ``None`` are ignored when creating the categories.
    """
    kwargs = {} if ordered is None else {'ordered': ordered}
    if isinstance(c, (pd.Series, pd.Categorical)):
        cats = c[~pd.isnull(c)].unique()
        if hasattr(cats, 'to_list'):
            cats = cats.to_list()
    elif hasattr(c, 'dtype'):
        cats = pd.unique(c[~pd.isnull(c)])
    else:
        cats = pd.unique([
            x for x, keep in zip(c, ~pd.isnull(c))
            if keep
        ])
    return pd.Categorical(c, categories=cats, **kwargs)


def _stable_series_sort(ser, ascending):
    """
    Stable sort for pandas series

    Temporary Solution until
        https://github.com/pandas-dev/pandas/issues/28697
        https://github.com/pandas-dev/pandas/pull/28698
    are resolved
    """
    from pandas.core.sorting import nargsort
    values = ser._values
    indexer = nargsort(
        values, kind='mergesort', ascending=ascending, na_position='last')
    return pd.Series(values[indexer], index=ser.index[indexer])
