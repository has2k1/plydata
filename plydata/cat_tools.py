"""
Functions for categoricals
"""
from itertools import chain, product

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from pandas.core.algorithms import value_counts

from .utils import last2

__all__ = [
    'cat_anon',
    'cat_collapse',
    'cat_concat',
    'cat_drop',
    'cat_expand',
    'cat_explicit_na',
    'cat_infreq',
    'cat_inorder',
    'cat_inseq',
    'cat_lump',
    'cat_lump_lowfreq',
    'cat_lump_min',
    'cat_lump_n',
    'cat_lump_prop',
    'cat_move',
    'cat_other',
    'cat_recode',
    'cat_relabel',
    'cat_relevel',
    'cat_rename',
    'cat_reorder',
    'cat_reorder2',
    'cat_rev',
    'cat_shift',
    'cat_shuffle',
    'cat_unify',
    'cat_zip',
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
    c = as_categorical(c)
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


def cat_move(c, *args, to=0):
    """
    Reorder categories explicitly

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    *args : tuple
        Categories to reorder. Any categories not mentioned
        will be left in existing order.
    to : int or inf
        Position where to place the categories. ``inf``, puts
        them at the end (highest value).

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> c = ['a', 'b', 'c', 'd', 'e']
    >>> cat_move(c, 'e', 'b')
    [a, b, c, d, e]
    Categories (5, object): [e, b, a, c, d]
    >>> cat_move(c, 'c', to=np.inf)
    [a, b, c, d, e]
    Categories (5, object): [a, b, d, e, c]
    >>> cat_move(pd.Categorical(c, ordered=True), 'a', 'c', 'e', to=1)
    [a, b, c, d, e]
    Categories (5, object): [b < a < c < e < d]
    """
    c = as_categorical(c)
    if np.isinf(to):
        to = len(c.categories)

    args = list(args)
    unmoved_cats = c.categories.drop(args).to_list()
    cats = unmoved_cats[0:to] + args + unmoved_cats[to:]
    c.reorder_categories(cats, inplace=True)
    return c


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
    c = as_categorical(c)
    c.reorder_categories(c.categories[::-1], inplace=True)
    return c


def cat_shift(c, n=1):
    """
    Shift and wrap-around categories to the left or right

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.

    n : int
        Number of times to shift. If positive, shift to
        the left, if negative shift to the right.
        Default is 1.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> c = ['a', 'b', 'c', 'd', 'e']
    >>> cat_shift(c)
    [a, b, c, d, e]
    Categories (5, object): [b, c, d, e, a]
    >>> cat_shift(c, 2)
    [a, b, c, d, e]
    Categories (5, object): [c, d, e, a, b]
    >>> cat_shift(c, -2)
    [a, b, c, d, e]
    Categories (5, object): [d, e, a, b, c]
    >>> cat_shift(pd.Categorical(c, ordered=True), -3)
    [a, b, c, d, e]
    Categories (5, object): [c < d < e < a < b]
    """
    c = as_categorical(c)
    cats = c.categories.to_list()
    cats_extended = cats + cats
    m = len(cats)
    n = n % m
    cats = cats_extended[n:m] + cats_extended[:n]
    c.reorder_categories(cats, inplace=True)
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
    c = as_categorical(c)
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


# Change the value of categories

def cat_anon(c, prefix='', random_state=None):
    """
    Anonymise categories

    Neither the value nor the order of the categories is preserved.

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
    >>> c = ['a', 'b', 'b', 'c', 'c', 'c']
    >>> cat_anon(c)
    [0, 1, 1, 2, 2, 2]
    Categories (3, object): [1, 0, 2]
    >>> cat_anon(c, 'c-', 321)
    [c-1, c-2, c-2, c-0, c-0, c-0]
    Categories (3, object): [c-0, c-2, c-1]
    >>> cat_anon(pd.Categorical(c, ordered=True), 'c-', 321)
    [c-1, c-2, c-2, c-0, c-0, c-0]
    Categories (3, object): [c-0 < c-2 < c-1]
    """
    c = as_categorical(c)
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.random.RandomState):
        raise TypeError(
            "Unknown type `{}` of random_state".format(type(random_state))
        )

    # Shuffle two times,
    #   1. to prevent predicable sequence to category mapping
    #   2. to prevent reversing of the new categories to the old ones
    fmt = '{}{}'.format
    cats = [fmt(prefix, i) for i in range(len(c.categories))]
    random_state.shuffle(cats)
    c.rename_categories(cats, inplace=True)
    cats = c.categories.to_list()
    random_state.shuffle(cats)
    c.reorder_categories(cats, inplace=True)
    return c


def cat_collapse(c, mapping, group_other=False):
    """
    Collapse categories into manually defined groups

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    mapping : dict
        New categories and the old categories contained in them.
    group_other : False
        If ``True``, a category is created to contain all other
        categories that have not been explicitly collapsed.
        The name of the other categories is ``other``, it may be
        postfixed by the first available integer starting from
        2 if there is a category with a similar name.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> c = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> mapping = {'first_2': ['a', 'b'], 'second_2': ['c', 'd']}
    >>> cat_collapse(c, mapping)
    [first_2, first_2, second_2, second_2, e, f]
    Categories (4, object): [first_2, second_2, e, f]
    >>> cat_collapse(c, mapping, group_other=True)
    [first_2, first_2, second_2, second_2, other, other]
    Categories (3, object): [first_2, second_2, other]

    Collapsing preserves the order

    >>> cat_rev(c)
    [a, b, c, d, e, f]
    Categories (6, object): [f, e, d, c, b, a]
    >>> cat_collapse(cat_rev(c), mapping)
    [first_2, first_2, second_2, second_2, e, f]
    Categories (4, object): [f, e, second_2, first_2]
    >>> mapping = {'other': ['a', 'b'], 'another': ['c', 'd']}
    >>> cat_collapse(c, mapping, group_other=True)
    [other, other, another, another, other2, other2]
    Categories (3, object): [other, another, other2]
    """
    def make_other_name():
        """
        Generate unique name for the other category
        """
        if 'other' not in mapping:
            return 'other'

        for i in range(2, len(mapping)+2):
            other = 'other' + str(i)
            if other not in mapping:
                return other

    c = as_categorical(c)
    if group_other:
        mapping = mapping.copy()
        other = make_other_name()
        mapped_categories = list(chain(*mapping.values()))
        unmapped_categories = c.categories.difference(mapped_categories)
        mapping[other] = list(unmapped_categories)

    inverted_mapping = {
        cat: new_cat
        for new_cat, old_cats in mapping.items()
        for cat in old_cats
    }

    # Convert old categories to new values in order and remove
    # any duplicates. The preserves the order
    new_cats = pd.unique([
        inverted_mapping.get(x, x)
        for x in c.categories
    ])

    c = pd.Categorical(
        [inverted_mapping.get(x, x) for x in c],
        categories=new_cats,
        ordered=c.ordered
    )
    return c


def cat_other(c, keep=None, drop=None, other_category='other'):
    """
    Replace categories with 'other'

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    keep : list-like
        Categories to preserve. Only one of ``keep`` or ``drop``
        should be specified.
    drop : list-like
        Categories to drop. Only one of ``keep`` or ``drop``
        should be specified.
    other_category : object
        Value used for the 'other' values. It is placed at
        the end of the categories.

    Returns
    -------
    out : categorical
        Values

    Examples
    --------
    >>> c = ['a', 'b', 'a', 'c', 'b', 'b', 'b', 'd', 'c']
    >>> cat_other(c, keep=['a', 'b'])
    [a, b, a, other, b, b, b, other, other]
    Categories (3, object): [a, b, other]
    >>> cat_other(c, drop=['a', 'b'])
    [other, other, other, c, other, other, other, d, c]
    Categories (3, object): [c, d, other]
    >>> cat_other(pd.Categorical(c, ordered=True), drop=['a', 'b'])
    [other, other, other, c, other, other, other, d, c]
    Categories (3, object): [c < d < other]
    """
    if keep is None and drop is None:
        raise ValueError(
            "Missing columns to `keep` or those to `drop`."
        )
    elif keep is not None and drop is not None:
        raise ValueError(
            "Only one of `keep` or `drop` should be given."
        )
    c = as_categorical(c)
    cats = c.categories

    if keep is not None:
        if not pdtypes.is_list_like(keep):
            keep = [keep]
    elif drop is not None:
        if not pdtypes.is_list_like(drop):
            drop = [drop]
        keep = cats.difference(drop)

    inverted_mapping = {
        cat: other_category
        for cat in cats.difference(keep)
    }
    inverted_mapping.update({x: x for x in keep})
    new_cats = cats.intersection(keep).to_list() + [other_category]
    c = pd.Categorical(
        [inverted_mapping.get(x, x) for x in c],
        categories=new_cats,
        ordered=c.ordered
    )
    return c


def _lump(lump_it, c, other_category):
    """
    Return a categorical of lumped

    Helper for cat_lump_* functions

    Parameters
    ----------
    lump_it : sequence[(obj, bool)]
        Sequence of (category, lump_category)
    c : cateorical
        Original categorical.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.

    Returns
    -------
    out : categorical
        Values
    """
    lookup = {
        cat: other_category if lump else cat
        for cat, lump in lump_it
    }
    new_cats = (
        c.categories
        .intersection(lookup.values())
        .insert(len(c), other_category)
    )

    c = pd.Categorical(
        [lookup[value] for value in c],
        categories=new_cats,
        ordered=c.ordered
    )
    return c


def cat_lump(
    c,
    n=None,
    prop=None,
    w=None,
    other_category='other',
    ties_method='min'
):
    """
    Lump together least or most common categories

    This is a general method that calls one of
    :func:`~plydata.cat_tools.cat_lump_n`
    :func:`~plydata.cat_tools.cat_lump_prop` or
    :func:`~plydata.cat_tools.cat_lump_lowfreq`
    depending on the parameters.

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    n : int (optional)
        Number of most/least common values to preserve (not lumped
        together). Positive ``n`` preserves the most common,
        negative ``n`` preserves the least common.

        Lumping happens on condition that the lumped category "other"
        will have the smallest number of items.
        You should only specify one of ``n`` or ``prop``
    prop : float (optional)
        Proportion above/below which the values of a category will be
        preserved (not lumped together). Positive ``prop`` preserves
        categories whose proportion of values is *more* than ``prop``.
        Negative ``prop`` preserves categories whose proportion of
        values is *less* than ``prop``.

        Lumping happens on condition that the lumped category "other"
        will have the smallest number of items.
        You should only specify one of ``n`` or ``prop``
    w : list[int|float] (optional)
        Weights for the frequency of each value. It should be the same
        length as ``c``.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.
    ties_method : {'min', 'max', 'average', 'first', 'dense'} (default: min)
        How to treat categories that occur the same number of times
        (i.e. ties):
        * min: lowest rank in the group
        * max: highest rank in the group
        * average: average rank of the group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups.

    Examples
    --------
    >>> cat_lump(list('abbccc'))
    [other, b, b, c, c, c]
    Categories (3, object): [b, c, other]

    When the least categories put together are not less than the next
    smallest group.

    >>> cat_lump(list('abcddd'))
    [a, b, c, d, d, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump(list('abcdddd'))
    [other, other, other, d, d, d, d]
    Categories (2, object): [d, other]

    >>> c = pd.Categorical(list('abccdd'))
    >>> cat_lump(c, n=1)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]

    >>> cat_lump(c, n=2)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]

    ``n`` Least common categories

    >>> cat_lump(c, n=-2)
    [a, b, other, other, other, other]
    Categories (3, object): [a, b, other]

    There are fewer than ``n`` categories that are the most/least common.

    >>> cat_lump(c, n=3)
    [a, b, c, c, d, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump(c, n=-3)
    [a, b, c, c, d, d]
    Categories (4, object): [a, b, c, d]

    By proportions, categories that make up *more* than ``prop`` fraction
    of the items.

    >>> cat_lump(c, prop=1/3.01)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]
    >>> cat_lump(c, prop=-1/3.01)
    [a, b, other, other, other, other]
    Categories (3, object): [a, b, other]
    >>> cat_lump(c, prop=1/2)
    [other, other, other, other, other, other]
    Categories (1, object): [other]

    Order of categoricals is maintained

    >>> c = pd.Categorical(
    ...     list('abccdd'),
    ...     categories=list('adcb'),
    ...     ordered=True
    ... )
    >>> cat_lump(c, n=2)
    [other, other, c, c, d, d]
    Categories (3, object): [d < c < other]

    **Weighted lumping**

    >>> c = list('abcd')
    >>> weights = [3, 2, 1, 1]
    >>> cat_lump(c, n=2)  # No lumping
    [a, b, c, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump(c, n=2, w=weights)
    [a, b, other, other]
    Categories (3, object): [a, b, other]
    """
    if n is not None:
        return cat_lump_n(c, n, w, other_category, ties_method)
    elif prop is not None:
        return cat_lump_prop(c, prop, w, other_category)
    else:
        return cat_lump_lowfreq(c, other_category)


def cat_lump_n(
    c,
    n,
    w=None,
    other_category='other',
    ties_method='min'
):
    """
    Lump together most/least common n categories

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    n : int
        Number of most/least common values to preserve (not lumped
        together). Positive ``n`` preserves the most common,
        negative ``n`` preserves the least common.

        Lumping happens on condition that the lumped category "other"
        will have the smallest number of items.
        You should only specify one of ``n`` or ``prop``
    w : list[int|float] (optional)
        Weights for the frequency of each value. It should be the same
        length as ``c``.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.
    ties_method : {'min', 'max', 'average', 'first', 'dense'} (default: min)
        How to treat categories that occur the same number of times
        (i.e. ties):
        * min: lowest rank in the group
        * max: highest rank in the group
        * average: average rank of the group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups.

    Examples
    --------
    >>> c = pd.Categorical(list('abccdd'))
    >>> cat_lump_n(c, 1)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]

    >>> cat_lump_n(c, 2)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]

    ``n`` Least common categories

    >>> cat_lump_n(c, -2)
    [a, b, other, other, other, other]
    Categories (3, object): [a, b, other]

    There are fewer than ``n`` categories that are the most/least common.

    >>> cat_lump_n(c, 3)
    [a, b, c, c, d, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump_n(c, -3)
    [a, b, c, c, d, d]
    Categories (4, object): [a, b, c, d]

    Order of categoricals is maintained

    >>> c = pd.Categorical(
    ...     list('abccdd'),
    ...     categories=list('adcb'),
    ...     ordered=True
    ... )
    >>> cat_lump_n(c, 2)
    [other, other, c, c, d, d]
    Categories (3, object): [d < c < other]

    **Weighted lumping**

    >>> c = list('abcd')
    >>> weights = [3, 2, 1, 1]
    >>> cat_lump_n(c, n=2)  # No lumping
    [a, b, c, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump_n(c, n=2, w=weights)
    [a, b, other, other]
    Categories (3, object): [a, b, other]
    """
    c = as_categorical(c)
    if len(c) == 0:
        return c

    if w is None:
        counts = c.value_counts().sort_values(ascending=False)
    else:
        counts = (
            pd.Series(w)
            .groupby(c)
            .apply(np.sum)
            .sort_values(ascending=False)
        )

    if n < 0:
        rank = counts.rank(method=ties_method)
        n = -n
    else:
        rank = (-counts).rank(method=ties_method)

    # Less than n categories outside the lumping,
    if not (rank > n).any():
        return c

    lump_it = zip(rank.index, rank > n)
    return _lump(lump_it, c, other_category)


def cat_lump_prop(
    c,
    prop,
    w=None,
    other_category='other',
):
    """
    Lump together least or most common categories by proportion

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    prop : float
        Proportion above/below which the values of a category will be
        preserved (not lumped together). Positive ``prop`` preserves
        categories whose proportion of values is *more* than ``prop``.
        Negative ``prop`` preserves categories whose proportion of
        values is *less* than ``prop``.

        Lumping happens on condition that the lumped category "other"
        will have the smallest number of items.
        You should only specify one of ``n`` or ``prop``
    w : list[int|float] (optional)
        Weights for the frequency of each value. It should be the same
        length as ``c``.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.

    Examples
    --------
    By proportions, categories that make up *more* than ``prop`` fraction
    of the items.

    >>> c = pd.Categorical(list('abccdd'))
    >>> cat_lump_prop(c, 1/3.01)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]
    >>> cat_lump_prop(c, -1/3.01)
    [a, b, other, other, other, other]
    Categories (3, object): [a, b, other]
    >>> cat_lump_prop(c, 1/2)
    [other, other, other, other, other, other]
    Categories (1, object): [other]
    """
    c = as_categorical(c)
    if len(c) == 0:
        return c

    if w is None:
        counts = c.value_counts().sort_values(ascending=False)
        total = len(c)
    else:
        counts = (
            pd.Series(w)
            .groupby(c)
            .apply(np.sum)
            .sort_values(ascending=False)
        )
        total = counts.sum()

    # For each category findout whether to lump it or keep it
    # Create a generator of the form ((cat, lump), ...)
    props = counts / total
    if prop < 0:
        if not (props > -prop).any():
            # No proportion more than target, so no lumping
            # the most common
            return c
        else:
            lump_it = zip(props.index, props > -prop)
    else:
        if not (props <= prop).any():
            # No proportion less than target, so no lumping
            # the least common
            return c
        else:
            lump_it = zip(props.index, props <= prop)

    return _lump(lump_it, c, other_category)


def cat_lump_lowfreq(
    c,
    other_category='other',
):
    """
    Lump together least categories

    Ensures that the "other" category is still the smallest.

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.

    Examples
    --------
    >>> cat_lump_lowfreq(list('abbccc'))
    [other, b, b, c, c, c]
    Categories (3, object): [b, c, other]

    When the least categories put together are not less than the next
    smallest group.

    >>> cat_lump_lowfreq(list('abcddd'))
    [a, b, c, d, d, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump_lowfreq(list('abcdddd'))
    [other, other, other, d, d, d, d]
    Categories (2, object): [d, other]
    """
    c = as_categorical(c)
    if len(c) == 0:
        return c

    # For each category findout whether to lump it or keep it
    # Create a generator of the form ((cat, lump), ...)
    counts = c.value_counts().sort_values(ascending=False)
    if len(counts) == 1:
        return c

    unique_counts = pd.unique(counts)
    smallest = unique_counts[-1]
    next_smallest = unique_counts[-2]
    smallest_counts = counts[counts == smallest]
    smallest_total = smallest_counts.sum()
    smallest_cats = smallest_counts.index

    if not smallest_total < next_smallest:
        return c

    lump_it = (
        (cat, True) if cat in smallest_cats else (cat, False)
        for cat in counts.index
    )
    return _lump(lump_it, c, other_category)


def cat_lump_min(
    c,
    min,
    w=None,
    other_category='other',
):
    """
    Lump catogeries, preserving those that appear min number of times

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    min : int
        Minum number of times a category must be represented to be
        preserved.
    w : list[int|float] (optional)
        Weights for the frequency of each value. It should be the same
        length as ``c``.
    other_category : object (default: 'other')
        Value used for the 'other' values. It is placed at
        the end of the categories.

    Examples
    --------
    >>> c = list('abccdd')
    >>> cat_lump_min(c, min=1)
    [a, b, c, c, d, d]
    Categories (4, object): [a, b, c, d]
    >>> cat_lump_min(c, min=2)
    [other, other, c, c, d, d]
    Categories (3, object): [c, d, other]

    **Weighted Lumping**

    >>> weights = [2, 2, .5, .5, 1, 1]
    >>> cat_lump_min(c, min=2, w=weights)
    [a, b, other, other, d, d]
    Categories (4, object): [a, b, d, other]

    Unlike :func:`~plydata.cat_tools.cat_lump`,  :func:`cat_lump_min`
    can lump together and create a category larger than the preserved
    categories.

    >>> c = list('abxyzccdd')
    >>> cat_lump_min(c, min=2)
    [other, other, other, other, other, c, c, d, d]
    Categories (3, object): [c, d, other]
    """
    c = as_categorical(c)
    if len(c) == 0:
        return c

    if w is None:
        counts = c.value_counts().sort_values(ascending=False)
    else:
        counts = (
            pd.Series(w)
            .groupby(c)
            .apply(np.sum)
            .sort_values(ascending=False)
        )

    if (counts >= min).all():
        return c

    lookup = {
        cat: cat if freq >= min else other_category
        for cat, freq in counts.items()
    }
    new_cats = (
        c.categories
        .intersection(lookup.values())
        .insert(len(c), other_category)
    )

    c = pd.Categorical(
        [lookup[value] for value in c],
        categories=new_cats,
        ordered=c.ordered
    )
    return c


def cat_rename(c, mapping=None, **kwargs):
    """
    Change/rename categories manually

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    mapping : dict (optional)
        Mapping of the form ``{old_name: new_name}`` for how to rename
        the categories. Setting a value to ``None`` removes the category.
        This arguments is useful if the old names are not valid
        python parameters. Otherwise, ``kwargs`` can be used.
    **kwargs : dict
        Mapping to rename categories. Setting a value to ``None`` removes
        the category.

    Examples
    --------
    >>> c = list('abcd')
    >>> cat_rename(c, a='A')
    [A, b, c, d]
    Categories (4, object): [A, b, c, d]
    >>> c = pd.Categorical(
    ...     list('abcd'),
    ...     categories=list('bacd'),
    ...     ordered=True
    ... )
    >>> cat_rename(c, b='B', d='D')
    [a, B, c, D]
    Categories (4, object): [B < a < c < D]

    Remove categories by setting them to ``None``.

    >>> cat_rename(c, b='B', d=None)
    [a, B, c]
    Categories (3, object): [B < a < c]
    """
    c = as_categorical(c)
    if mapping is not None and len(kwargs):
        raise ValueError("Use only one of `new` or the ``kwargs``.")

    lookup = mapping or kwargs

    if not lookup:
        return c

    # Remove categories set to None
    remove = [
        old
        for old, new in lookup.items()
        if new is None
    ]
    if remove:
        for cat in remove:
            del lookup[cat]
        c = c.remove_categories(remove).dropna()

    # Separately change values (inplace) and the categories (using an
    # array) old to the new names. Then reconcile the two lists.
    categories = c.categories.to_numpy().copy()
    c.add_categories(
        pd.Index(lookup.values()).difference(c.categories),
        inplace=True
    )
    for old, new in lookup.items():
        if old not in c.categories:
            raise IndexError("Unknown category '{}'.".format(old))
        c[c == old] = new
        categories[categories == old] = new

    new_categories = pd.unique(categories)
    c.remove_unused_categories(inplace=True)
    c.set_categories(new_categories, inplace=True)
    return c


def cat_relabel(c, func=None, *args, **kwargs):
    """
    Change/rename categories and collapse as necessary

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    func : callable
        Function to create the new name. The first argument to
        the function will be a category to be renamed.
    *args : tuple
        Positional arguments passed to ``func``.
    *kwargs : dict
        Keyword arguments passed to ``func``.

    Examples
    --------
    >>> c = list('abcde')
    >>> cat_relabel(c, str.upper)
    [A, B, C, D, E]
    Categories (5, object): [A, B, C, D, E]
    >>> c = pd.Categorical([0, 1, 2, 1, 1, 0])
    >>> def func(x):
    ...     if x == 0:
    ...         return 'low'
    ...     elif x == 1:
    ...         return 'mid'
    ...     elif x == 2:
    ...         return 'high'
    >>> cat_relabel(c, func)
    [low, mid, high, mid, mid, low]
    Categories (3, object): [low, mid, high]

    When the function yields the same output for 2 or more
    different categories, those categories are collapsed.

    >>> def first(x):
    ...     return x[0]
    >>> c = pd.Categorical(['aA', 'bB', 'aC', 'dD'],
    ...     categories=['bB', 'aA', 'dD', 'aC'],
    ...     ordered=True
    ... )
    >>> cat_relabel(c, first)
    [a, b, a, d]
    Categories (3, object): [b < a < d]
    """
    c = as_categorical(c)
    new_categories = [func(x, *args, **kwargs) for x in c.categories]
    new_categories_uniq = pd.unique(new_categories)
    if len(new_categories_uniq) < len(c.categories):
        # Collapse
        lookup = dict(zip(c.categories, new_categories))
        c = pd.Categorical(
            [lookup[value] for value in c],
            categories=new_categories_uniq,
            ordered=c.ordered
        )
    else:
        c.categories = new_categories

    return c


def cat_expand(c, *args):
    """
    Add additional categories to a categorical

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    *args : tuple
        Categories to add.

    Examples
    --------
    >>> cat_expand(list('abc'), 'd', 'e')
    [a, b, c]
    Categories (5, object): [a, b, c, d, e]
    >>> c = pd.Categorical(list('abcd'), ordered=True)
    >>> cat_expand(c, 'e', 'f')
    [a, b, c, d]
    Categories (6, object): [a < b < c < d < e < f]
    """
    c = as_categorical(c)
    c.add_categories(
        pd.Index(args).difference(c.categories),
        inplace=True
    )
    return c


def cat_explicit_na(c, na_category='(missing)'):
    """
    Give missing values an explicity category

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    na_category : object (default: '(missing)')
        Category for missing values

    Examples
    --------
    >>> c = pd.Categorical(
    ...     ['a', 'b', None, 'c', None, 'd', 'd'],
    ...     ordered=True
    ... )
    >>> c
    [a, b, NaN, c, NaN, d, d]
    Categories (4, object): [a < b < c < d]
    >>> cat_explicit_na(c)
    [a, b, (missing), c, (missing), d, d]
    Categories (5, object): [a < b < c < d < (missing)]
    """
    c = as_categorical(c)
    bool_idx = pd.isnull(c)
    if any(bool_idx):
        c.add_categories([na_category], inplace=True)
        c[bool_idx] = na_category
    return c


def cat_remove_unused(c, only=None):
    """
    Remove unused categories

    Parameters
    ----------
    c : list-like
        Values that will make up the categorical.
    only : list-like (optional)
        The categories to remove *if* they are empty. If not given,
        all unused categories are dropped.

    Examples
    --------
    >>> c = pd.Categorical(list('abcdd'), categories=list('bacdefg'))
    >>> c
    [a, b, c, d, d]
    Categories (7, object): [b, a, c, d, e, f, g]
    >>> cat_remove_unused(c)
    [a, b, c, d, d]
    Categories (4, object): [b, a, c, d]
    >>> cat_remove_unused(c, only=['a', 'e', 'g'])
    [a, b, c, d, d]
    Categories (5, object): [b, a, c, d, f]
    """
    if not pdtypes.is_categorical(c):
        # All categories are used
        c = pd.Categorical(c)
        return c
    else:
        c = c.copy()

    if only is None:
        only = c.categories

    used_idx = pd.unique(c.codes)
    used_categories = c.categories[used_idx]
    c = c.remove_categories(
        c.categories
        .difference(used_categories)
        .intersection(only)
    )
    return c


def cat_unify(cs, categories=None):
    """
    Unify (union of all) the categories in a list of categoricals

    Parameters
    ----------
    cs : list-like
        Categoricals
    categories : list-like
        Extra categories to apply to very categorical.

    Examples
    --------
    >>> c1 = pd.Categorical(['a', 'b'], categories=list('abc'))
    >>> c2 = pd.Categorical(['d', 'e'], categories=list('edf'))
    >>> c1_new, c2_new = cat_unify([c1, c2])
    >>> c1_new
    [a, b]
    Categories (6, object): [a, b, c, e, d, f]
    >>> c2_new
    [d, e]
    Categories (6, object): [a, b, c, e, d, f]
    >>> c1_new, c2_new = cat_unify([c1, c2], categories=['z', 'y'])
    >>> c1_new
    [a, b]
    Categories (8, object): [a, b, c, e, d, f, z, y]
    >>> c2_new
    [d, e]
    Categories (8, object): [a, b, c, e, d, f, z, y]
    """
    cs = [as_categorical(c) for c in cs]
    all_cats = list(chain(*(c.categories.to_list() for c in cs)))
    if categories is None:
        categories = pd.unique(all_cats)
    else:
        categories = pd.unique(all_cats + categories)

    cs = [c.set_categories(categories) for c in cs]
    return cs


def cat_concat(*args):
    """
    Concatenate categoricals and combine the categories

    Parameters
    ----------
    *args : tuple
        Categoricals to be concatenated

    Examples
    --------
    >>> c1 = pd.Categorical(['a', 'b'], categories=['b', 'a'])
    >>> c2 = pd.Categorical(['d', 'a', 'c'])
    >>> cat_concat(c1, c2)
    [a, b, d, a, c]
    Categories (4, object): [b, a, c, d]

    Notes
    -----
    The resulting category is not ordered.
    """
    categories = pd.unique(list(chain(*(
        c.categories if pdtypes.is_categorical(c) else c
        for c in args
    ))))
    cs = pd.Categorical(
        list(chain(*(c for c in args))),
        categories=categories
    )
    return cs


def cat_zip(*args, sep=':', keep_empty=False):
    """
    Create a new categorical (zip style) combined from two or more

    Parameters
    ----------
    *args : tuple
        Categoricals to be concatenated.
    sep : str (default: ':')
        Separator for the combined categories.
    keep_empty : bool (default: False)
        If ``True``, include all combinations of categories
        even those without observations.

    Examples
    --------
    >>> c1 = pd.Categorical(list('aba'))
    >>> c2 = pd.Categorical(list('122'))
    >>> cat_zip(c1, c2)
    [a:1, b:2, a:2]
    Categories (3, object): [a:1, a:2, b:2]
    >>> cat_zip(c1, c2, keep_empty=True)
    [a:1, b:2, a:2]
    Categories (4, object): [a:1, a:2, b:1, b:2]
    """
    values = [sep.join(items) for items in zip(*args)]
    cs = [
        c if pdtypes.is_categorical(c) else pd.Categorical(c)
        for c in args
    ]
    categories = [
        sep.join(items)
        for items in product(*(c.categories for c in cs))
    ]

    c = pd.Categorical(values, categories=categories)

    if not keep_empty:
        c.remove_unused_categories(inplace=True)

    return c


# helpers
def as_categorical(c, copy=True):
    """
    Convert input to a categorical

    Parameters
    ----------
    c : categorical_like
        Sequence of objects
    copy : bool
        If `True` and c is alread a categorical, return
        a copy of `c` otherwise return `c`.

    Returns
    -------
    out : categorical
        Categorical made out of `c` or copy of `c`
        if it was a categorical

    """
    if not pdtypes.is_categorical(c):
        c = pd.Categorical(c)
    elif copy:
        c = c.copy()
    return c


# Temporary functions

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


cat_relevel = cat_move
cat_recode = cat_rename
cat_drop = cat_remove_unused
