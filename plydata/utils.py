import re
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .eval import EvalEnvironment
from .options import options

BOOL_PATTERN = re.compile(r'True|False')


def hasattrs(obj, names):
    """
    Return True of obj has all the names attributes
    """
    return all(hasattr(obj, attr) for attr in names)


@contextmanager
def temporary_key(d, key, value):
    """
    Context manager that removes key from dictionary on closing

    The dictionary will hold the key for the duration of
    the context.

    Parameters
    ----------
    d : dict-like
        Dictionary in which to insert a temporary key.
    key : hashable
        Location at which to insert ``value``.
    value : object
        Value to insert in ``d`` at location ``key``.
    """
    d[key] = value
    try:
        yield d
    finally:
        del d[key]


@contextmanager
def temporary_attr(obj, name, value):
    """
    Context manager that removes key from dictionary on closing

    The dictionary will hold the key for the duration of
    the context.

    Parameters
    ----------
    obj : object
        Object onto which to add a temporary attribute.
    name : str
        Name of attribute to add to ``obj``.
    value : object
        Value of ``attr``.
    """
    setattr(obj, name, value)
    try:
        yield obj
    finally:
        delattr(obj, name)


def get_empty_env():
    """
    Return an empty environment

    This is for testing or documentation purposes
    """
    return EvalEnvironment(namespaces={})


def Q(name):
    """
    Quote a variable name

    A way to 'quote' variable names, especially ones that do not otherwise
    meet Python's variable name rules.

    Parameters
    ----------
    name : str
        Name of variable

    Returns
    -------
    value : object
        Value of variable

    Examples
    --------
    >>> import pandas as pd
    >>> from plydata import define
    >>> df = pd.DataFrame({'class': [10, 20, 30]})

    Since ``class`` is a reserved python keyword it cannot be a variable
    name, and therefore cannot be used in an expression without quoting it.

    >>> df >> define(y='class+1')
    Traceback (most recent call last):
      File "<string>", line 1
        class+1
            ^
    SyntaxError: invalid syntax

    >>> df >> define(y='Q("class")+1')
       class   y
    0     10  11
    1     20  21
    2     30  31

    Note that it is ``'Q("some name")'`` and not ``'Q(some name)'``.
    As in the above example, you do not need to ``import`` ``Q`` before
    you can use it.
    """
    env = EvalEnvironment.capture(1)
    try:
        return env.namespace[name]
    except KeyError:
        raise NameError("No data named {!r} found".format(name))


def n():
    """
    Size of a group

    It can be used in verbs like
    :class:`~plydata.one_table_verbs.summarize`,
    :class:`~plydata.one_table_verbs.define`. and
    :class:`~plydata.one_table_verbs.create`.
    This is special function that is internally created for each
    group dataframe.
    """
    # For documentation purposes


class custom_dict(dict):
    """
    Dict datastore for conflict testing purposes

    Using a regular dict creates conflicts with verbs
    whose first parameter can be a dict
    """
    pass


@contextmanager
def regular_index(*dfs):
    """
    Change & restore the indices of dataframes

    Dataframe with duplicate values can be hard to work with.
    When split and recombined, you cannot restore the row order.
    This can be the case even if the index has unique but
    irregular/unordered. This contextmanager resets the unordered
    indices of any dataframe passed to it, on exit it restores
    the original index.

    A regular index is of the form::

        RangeIndex(start=0, stop=n, step=1)

    Parameters
    ----------
    dfs : tuple
        Dataframes

    Yields
    ------
    dfs : tuple
        Dataframe

    Examples
    --------
    Create dataframes with different indices

    >>> df1 = pd.DataFrame([4, 3, 2, 1])
    >>> df2 = pd.DataFrame([3, 2, 1], index=[3, 0, 0])
    >>> df3 = pd.DataFrame([11, 12, 13], index=[11, 12, 13])

    Within the contexmanager all frames have nice range indices

    >>> with regular_index(df1, df2, df3):
    ...     print(df1.index)
    ...     print(df2.index)
    ...     print(df3.index)
    RangeIndex(start=0, stop=4, step=1)
    RangeIndex(start=0, stop=3, step=1)
    RangeIndex(start=0, stop=3, step=1)

    Indices restored

    >>> df1.index
    RangeIndex(start=0, stop=4, step=1)
    >>> df2.index
    Int64Index([3, 0, 0], dtype='int64')
    >>> df3.index
    Int64Index([11, 12, 13], dtype='int64')
    """
    original_index = [df.index for df in dfs]
    have_bad_index = [not isinstance(df.index, pd.RangeIndex)
                      for df in dfs]

    for df, bad in zip(dfs, have_bad_index):
        if bad:
            df.reset_index(drop=True, inplace=True)

    try:
        yield dfs
    finally:
        for df, bad, idx in zip(dfs, have_bad_index, original_index):
            if bad and len(df.index) == len(idx):
                df.index = idx


def unique(lst):
    """
    Return unique elements

    :class:`pandas.unique` and :class:`numpy.unique` cast
    mixed type lists to the same type. They are faster, but
    some times we want to maintain the type.

    Parameters
    ----------
    lst : list-like
        List of items

    Returns
    -------
    out : list
        Unique items in the order that they appear in the
        input.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> lst = ['one', 'two', 123, 'three']
    >>> pd.unique(lst)
    array(['one', 'two', '123', 'three'], dtype='<U5')
    >>> np.unique(lst)
    array(['123', 'one', 'three', 'two'],
          dtype='<U5')
    >>> unique(lst)
    ['one', 'two', 123, 'three']

    pandas and numpy cast 123 to a string!, and numpy does not
    even maintain the order.
    """
    seen = set()

    def make_seen(x):
        seen.add(x)
        return x

    return [make_seen(x) for x in lst if x not in seen]


def identity(*args):
    """
    Return whatever is passed in

    Examples
    --------
    >>> x = 1
    >>> y = 2
    >>> identity(x)
    1
    >>> identity(x, y)
    (1, 2)
    >>> identity(*(x, y))
    (1, 2)
    """
    return args if len(args) > 1 else args[0]


def clean_indices(df, sep='_', inplace=False):
    """
    Clearup any multi/fancy indices

    1. columns multiindices are flattened
    2. Fancy multivariable row indices are turned into
       columns and the row index set regular form (0..n)

    Parameters
    ----------
    df : dataframe
        Dataframe
    sep : str
        Separator for the new column names

    Returns
    -------
    out : dataframe
        Dataframe

    Examples
    --------
    >>> import pandas as pd
    >>> ridx = pd.MultiIndex.from_tuples(
    ...     [(1, 'red'), (1, 'blue'),
    ...      (2, 'red'), (2, 'blue')],
    ...     names=('number', 'color')
    ... )
    >>> cidx = pd.MultiIndex.from_product(
    ...     [['part1', 'part2'], ['numeric', 'char']],
    ...     names=('parts','types')
    ... )
    >>> df = pd.DataFrame({
    ...     'w': [1, 2, 3, 4],
    ...     'x': list('aabb'),
    ...     'y': [5, 6, 7, 8],
    ...     'z': list('ccdd')
    ...     }, index=ridx
    ... )
    >>> df.columns = cidx
    >>> df
    parts          part1        part2
    types        numeric char numeric char
    number color
    1      red         1    a       5    c
           blue        2    a       6    c
    2      red         3    b       7    d
           blue        4    b       8    d
    >>> clean_indices(df)
       number color  part1_numeric part1_char  part2_numeric part2_char
    0       1   red              1          a              5          c
    1       1  blue              2          a              6          c
    2       2   red              3          b              7          d
    3       2  blue              4          b              8          d

    When the inner levels are unique, the names are not joined

    >>> cidx2 = pd.MultiIndex.from_tuples(
    ...     [('part1', 'numeric1'), ('part1', 'char1'),
    ...      ('part2', 'numeric2'), ('part2', 'char2')],
    ...     names=('parts','types')
    ... )
    >>> df.columns = cidx2
    >>> df
    parts           part1          part2
    types        numeric1 char1 numeric2 char2
    number color
    1      red          1     a        5     c
           blue         2     a        6     c
    2      red          3     b        7     d
           blue         4     b        8     d
    >>> clean_indices(df)
       number color  numeric1 char1  numeric2 char2
    0       1   red         1     a         5     c
    1       1  blue         2     a         6     c
    2       2   red         3     b         7     d
    3       2  blue         4     b         8     d
    """
    if not inplace:
        df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = collapse_multiindex(df.columns, sep)

    df.reset_index(inplace=True)
    df.columns.name = None
    df.index.name = None
    return None if inplace else df


def collapse_multiindex(midx, sep='_'):
    """
    Collapse a MultiIndex into a minimal Index

    Parameters
    ----------
    midx : pandas.MultiIndex
        MultiIndex to be collapsed

    Returns
    -------
    out : pandas.Index
        Flat Index

    Examples
    --------
    >>> m1 = pd.MultiIndex.from_product([list('a'), list('12')])
    >>> m1
    MultiIndex([('a', '1'),
                ('a', '2')],
               )
    >>> collapse_multiindex(m1)
    Index(['1', '2'], dtype='object')
    >>> m2 = pd.MultiIndex.from_product([list('ab'), list('12')])
    >>> m2
    MultiIndex([('a', '1'),
                ('a', '2'),
                ('b', '1'),
                ('b', '2')],
               )
    >>> collapse_multiindex(m2)
    Index(['a_1', 'a_2', 'b_1', 'b_2'], dtype='object')
    >>> m3 = pd.MultiIndex.from_tuples(
    ...     [('a', '1'), ('a', '2'),
    ...      ('b', '1'), ('b', '1')]
    ... )
    >>> m3
    MultiIndex([('a', '1'),
                ('a', '2'),
                ('b', '1'),
                ('b', '1')],
               )
    >>> collapse_multiindex(m3)
    Traceback (most recent call last):
        ...
    ValueError: Cannot create unique column names.
    """
    def is_unique(lst):
        return len(set(lst)) == len(lst)

    def make_name(toks):
        if len(toks) == 1:
            # Preserves integer column names for basic
            # simple case when they will not be joined up
            # with another name up the hierarchy
            return toks[0]
        else:
            return sep.join(str(t) for t in toks)

    # Minimum tokens required to uniquely identify columns.
    # We start with the columns in the inner most level of
    # the multiindex.
    # - [(a, 1), (a, 2)] -> [(1,), (2,)]
    # - [(a, 1), (a, 2), (b, 1), (b, 2)] ->
    #       [(a, 1), (a, 2), (b, 1), (b, 2)]
    # - [(z, a, 1), (z, a, 2), (z, b, 1), (z, b, 2)] ->
    #       [(a, 1), (a, 2), (b, 1), (b, 2)]
    for i in range(midx.nlevels):
        id_tokens = [x[-(1+i):] for x in midx]
        if is_unique(id_tokens):
            break
    else:
        raise ValueError("Cannot create unique column names.")

    columns = [make_name(toks) for toks in id_tokens]
    return pd.Index(columns)


def convert_str(data, columns=None):
    """
    Try converting string/object columns in data to more specific dtype

    This function modifies the input data.

    Parameters
    ----------
    data : dataframe
        Data
    columns : list-like or None
        Names of columns to check and maybe convert.
        If ``None``, all the string columns are converted.

    Returns
    -------
    data : dataframe
        Data
    """
    if columns is None:
        columns = [
            name
            for name, col in data.items()
            if hasattr(col, 'str')
        ]

    def is_numeric(col):
        return col.str.isnumeric().all()

    def is_float(col):
        try:
            col.astype(float)
        except ValueError:
            return False
        else:
            return True

    def is_bool(col):
        return col.str.match(BOOL_PATTERN).all()

    for name in columns:
        col = data[name]

        if is_numeric(col) or is_float(col):
            data[name] = pd.to_numeric(col)
        elif is_bool(col):
            data[name] = col.replace({
                'True': True,
                'False': False
            })

    return data


def verify_arg(value, name, options):
    """
    Verify Argument

    Parameter
    ---------
    value : int | str
        Value of argument
    name : str
        Name of argument
    options : list-like | set
        Allowed values of argument

    Raises
    ------
    ValueError
        If value is not in the allowed options.

    Examples
    --------
    >>> verify_arg('dog', 'pet', ('fish', 'dog', 'cat'))
    >>> verify_arg('snail', 'pet', ('fish', 'dog', 'cat'))
    Traceback (most recent call last):
        ...
    ValueError: Got pet='snail'. Should be one of ('dog', 'fish', 'cat')
    """
    if value not in options:
        raise ValueError(
            "Got {}={!r}. Should be one of {!r}".format(
                name, value, options
            )
        )


def mean_if_many(x):
    """
    Compute mean of x if x has more than 1 element

    If x has one element, return that element.
    By only computing the mean if x is greater than 1;

        - singular integer values remain integers
        - a single string value passes through so this can be used as
          an aggregate function (aggfunc) when pivoting. This avoids an
          unnecessary error.

    Parameters
    ----------
    x : list-like
        Values whose mean to compute

    Returns
    -------
    out : object
        Mean of x or the only value in x

    Examples
    --------
    >>> mean_if_many([4])
    4
    >>> mean_if_many([4, 4])
    4.0
    >>> mean_if_many([4, 5, 6, 7])
    5.5
    >>> mean_if_many(['string_1'])
    'string_1'
    >>> mean_if_many(['string_1', 'string_2'])
    Traceback (most recent call last):
        ...
    TypeError: cannot perform reduce with flexible type
    """
    return list(x)[0] if len(x) == 1 else np.mean(x)


def last2(x, y):
    """
    Find last value of y when sorted by x

    Parameters
    ----------
    x : list-like
        Values
    y : list-like
        Values

    Returns
    -------
    obj : object
        Last value of y when sorted by x

    Examples
    --------
    >>> x = [1, 2, 3, 99, 5, 6]
    >>> y = [1, 2, 3, 4, 5, 6]
    >>> last2(x, y)
    4
    >>> last2(x, y[::-1])
    3

    See Also
    --------
    :class:`~plydata.cat_tools.reorder2`
    """
    y = np.asarray(y)
    return y[np.argsort(x)][-1]


def first2(x, y):
    """
    Find first value of y when sorted by x

    Parameters
    ----------
    x : list-like
        Values
    y : list-like
        Values

    Returns
    -------
    obj : object
        Last value of y when sorted by x

    Examples
    --------
    >>> x = [1, 2, 3, -99, 5, 6]
    >>> y = [1, 2, 3, 4, 5, 6]
    >>> first2(x, y)
    4
    >>> first2(x, y[::-1])
    3

    See Also
    --------
    :class:`~plydata.cat_tools.reorder2`
    """
    y = np.asarray(y)
    return y[np.argsort(x)][0]


def ply(data, *verbs):
    """
    Pipe data through the verbs

    This function allows you to use plydata without
    abusing the ``>>`` operator.

    Parameters
    ----------
    data : dataframe
        Data
    verbs : tuple
        Verb to which the data should be piped

    >>> from plydata import *
    >>> df = pd.DataFrame({
    ...     'x': [0, 1, 2, 3],
    ...     'y': ['zero', 'one', 'two', 'three']}
    ... )

    Using ply

    >>> ply(
    ...    df,
    ...    define(z='2*x', w='y+"-"+y'),
    ...    group_by(parity='x % 2'),
    ...    define(u='sum(z)')
    ... )
    groups: ['parity']
       x      y  z            w  parity  u
    0  0   zero  0    zero-zero       0  4
    1  1    one  2      one-one       1  8
    2  2    two  4      two-two       0  4
    3  3  three  6  three-three       1  8

    Is equivalent to

    >>> (df
    ...  >> define(z='2*x', w='y+"-"+y')
    ...  >> group_by(parity='x % 2')
    ...  >> define(u='sum(z)'))
    groups: ['parity']
       x      y  z            w  parity  u
    0  0   zero  0    zero-zero       0  4
    1  1    one  2      one-one       1  8
    2  2    two  4      two-two       0  4
    3  3  three  6  three-three       1  8
    """
    data = data.copy()
    with options(modify_input_data=True):
        for verb in verbs:
            data >>= verb
    return data
