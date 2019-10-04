"""
Functions for categoricals
"""
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from pandas.core.algorithms import value_counts

from .utils import last2

__all__ = [
    'cat_infreq',
    'cat_inorder',
    'cat_inseq',
    'cat_reorder',
    'cat_reorder2',
    'cat_rev',
    'cat_shuffle',
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


def cat_inseq(c, ordered=None):
    """
    Reorder categorical by numerical order

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
    >>> x = pd.Categorical([5, 1, 3, 2, 4])
    >>> cat_inseq(x)
    [5, 1, 3, 2, 4]
    Categories (5, int64): [1, 2, 3, 4, 5]
    >>> x = pd.Categorical([5, 1, '3', 2, 4])
    >>> cat_inseq(x)
    [5, 1, 3, 2, 4]
    Categories (5, int64): [1, 2, 3, 4, 5]

    Values that cannot be coerced to numerical turn in ``NaN``,
    and categories cannot be ``NaN``.

    >>> x = pd.Categorical([5, 1, 'three', 2, 4])
    >>> cat_inseq(x)
    [5, 1, NaN, 2, 4]
    Categories (4, int64): [1, 2, 4, 5]

    Coerces values to numerical

    >>> x = [5, 1, '3', 2, 4]
    >>> cat_inseq(x, ordered=True)
    [5, 1, 3, 2, 4]
    Categories (5, int64): [1 < 2 < 3 < 4 < 5]
    >>> x = [5, 1, '3', 2, '4.5']
    >>> cat_inseq(x)
    [5.0, 1.0, 3.0, 2.0, 4.5]
    Categories (5, float64): [1.0, 2.0, 3.0, 4.5, 5.0]

    Atleast one of the values must be coercible to the integer

    >>> x = ['five', 'one', 'three', 'two', 'four']
    >>> cat_inseq(x)
    Traceback (most recent call last):
        ...
    ValueError: Atleast one existing category must be a number.
    >>> x = ['five', 'one', '3', 'two', 'four']
    >>> cat_inseq(x)
    [NaN, NaN, 3, NaN, NaN]
    Categories (1, int64): [3]
    """
    if not isinstance(c, pd.Categorical):
        c = pd.Categorical(c)
    else:
        c = c.copy()

    # one value at a time to avoid turning integers into floats
    # when some values create nans
    numerical_cats = []
    for x in c.categories:
        _x = pd.to_numeric(x, 'coerce')
        if not pd.isnull(_x):
            numerical_cats.append(_x)

    if len(numerical_cats) == 0 and len(c) > 0:
        raise ValueError(
            "Atleast one existing category must be a number."
        )

    # Change the original categories to numerical ones, making sure
    # to rename the existing ones i.e '3' becomes 3. Only after that,
    # change to order.
    c = (c.set_categories(numerical_cats, rename=True)
         .reorder_categories(sorted(numerical_cats)))

    if ordered is not None:
        c.set_ordered(ordered, inplace=True)
    return c


def cat_reorder(c, x, fun=np.median, ascending=True):
    """
    Reorder categorical by sorting along another variable

    It is the order of the categories that changes. Values in x
    are grouped by categories and summarised to determine the
    new order.

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    x : list-like
        Values by which ``c`` will be ordered.
    fun : callable
        Summarising function to ``x`` for each category in ``c``.
        Default is the *median*.
    ascending : bool
        If ``True``, the ``c`` is ordered in ascending order of ``x``.

    Examples
    --------
    >>> c = list('abbccc')
    >>> x = [11, 2, 2, 3, 33, 3]
    >>> cat_reorder(c, x)
    [a, b, b, c, c, c]
    Categories (3, object): [b, c, a]
    >>> cat_reorder(c, x, fun=max)
    [a, b, b, c, c, c]
    Categories (3, object): [b, a, c]
    >>> cat_reorder(c, x, fun=max, ascending=False)
    [a, b, b, c, c, c]
    Categories (3, object): [c, a, b]
    >>> c_ordered = pd.Categorical(c, ordered=True)
    >>> cat_reorder(c_ordered, x)
    [a, b, b, c, c, c]
    Categories (3, object): [b < c < a]
    >>> cat_reorder(c + ['d'], x)
    Traceback (most recent call last):
        ...
    ValueError: Lengths are not equal. len(c) is 7 and len(x) is 6.
    """
    if len(c) != len(x):
        raise ValueError(
            "Lengths are not equal. len(c) is {} and len(x) is {}.".format(
                len(c), len(x)
            )
        )
    summary = (pd.Series(x)
               .groupby(c)
               .apply(fun)
               .sort_values(ascending=ascending)
               )
    cats = summary.index.to_list()
    return pd.Categorical(c, categories=cats)


def cat_reorder2(c, x, y, *args, fun=last2, ascending=False, **kwargs):
    """
    Reorder categorical by sorting along another variable

    It is the order of the categories that changes. Values in x
    are grouped by categories and summarised to determine the
    new order.

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    x : list-like
        Values by which ``c`` will be ordered.
    y : list-like
        Values by which ``c`` will be ordered.
    *args : tuple
        Position arguments passed to function fun.
    fun : callable
        Summarising function to ``x`` for each category in ``c``.
        Default is the *median*.
    ascending : bool
        If ``True``, the ``c`` is ordered in ascending order of ``x``.
    **kwargs : dict
        Keyword arguments passed to ``fun``.

    Examples
    --------

    Order stocks by the price in the latest year. This type of ordering
    can be used to order line plots so that the ends match the order of
    the legend.

    >>> stocks = list('AAABBBCCC')
    >>> year = [1980, 1990, 2000] * 3
    >>> price = [12.34, 12.90, 13.55, 10.92, 14.73, 11.08, 9.02, 12.44, 15.65]
    >>> cat_reorder2(stocks, year, price)
    [A, A, A, B, B, B, C, C, C]
    Categories (3, object): [C, A, B]
    """
    if len(c) != len(x) or len(x) != len(y):
        raise ValueError(
            "Lengths are not equal. len(c) is {}, len(x) is {} and "
            "len(y) is {}.".format(len(c), len(x), len(y))
        )

    # Wrap two argument function fun with a function that
    # takes a dataframe, put x and y into a dataframe, then
    # use dataframe.groupby
    def _fun(cat_df):
        return fun(cat_df['x'], cat_df['y'], *args, **kwargs)

    summary = (pd.DataFrame({'x': x, 'y': y})
               .groupby(c)
               .apply(_fun)
               .sort_values(ascending=ascending)
               )
    cats = summary.index.to_list()
    return pd.Categorical(c, categories=cats)


def cat_rev(c):
    """
    Reverse order of categories

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> c = ['a', 'b', 'c']
    >>> cat_rev(c)
    [a, b, c]
    Categories (3, object): [c, b, a]
    >>> cat_rev(pd.Categorical(c))
    [a, b, c]
    Categories (3, object): [c, b, a]
    """
    if not pdtypes.is_categorical(c):
        c = pd.Categorical(c)
    else:
        c = c.copy()
    c.reorder_categories(c.categories[::-1], inplace=True)
    return c


def cat_shuffle(c, random_state=None):
    """
    Reverse order of categories

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    random_state : int or ~numpy.random.RandomState, optional
        Seed or Random number generator to use. If ``None``, then
        numpy global generator :class:`numpy.random` is used.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> np.random.seed(123)
    >>> c = ['a', 'b', 'c', 'd', 'e']
    >>> cat_shuffle(c)
    [a, b, c, d, e]
    Categories (5, object): [b, d, e, a, c]
    >>> cat_shuffle(pd.Categorical(c, ordered=True), 321)
    [a, b, c, d, e]
    Categories (5, object): [d < b < a < c < e]
    """
    if not pdtypes.is_categorical(c):
        c = pd.Categorical(c)
    else:
        c = c.copy()

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.random.RandomState):
        raise TypeError(
            "Unknown type `{}` of random_state".format(type(random_state))
        )

    cats = c.categories.to_list()
    random_state.shuffle(cats)
    c.reorder_categories(cats, inplace=True)
    return c
    else:
        raise TypeError(
            "Unknown type `{}` of random_state".format(type(random_state))
        )

    cats = c.categories.to_list()
    random_state.shuffle(cats)
    c.reorder_categories(cats, inplace=True)
    return c


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
