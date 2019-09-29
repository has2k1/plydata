"""
Functions for categoricals
"""

import pandas as pd

__all__ = ['cat_inorder']


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
