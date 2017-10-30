from contextlib import contextmanager
import keyword

import pandas as pd

from .eval import EvalEnvironment

KEYWORDS = set(keyword.kwlist)


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
    array(['one', 'two', '123', 'three'], dtype=object)
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


class Expression:
    """
    An expression that will be evaluated

    Parameters
    ----------
    stmt : str or function
        Statement that will be evaluated. Some verbs
        allow only one or the other.
    column : str
        Column in which the result of the statment
        will be placed.
    """
    stmt = None
    column = None

    def __init__(self, stmt, column):
        self.stmt = stmt
        self.column = column

    def __str__(self):
        return 'Expression({!r}, {})'.format(self.stmt, self.column)

    def __repr__(self):
        return str(self)

    def evaluate(self, data, env):
        """
        Evaluate statement

        Parameters
        ----------
        data : pandas.DataFrame
            Data in whose namespace the statement will be
            evaluated. Typically, this is a group dataframe.

        Returns
        -------
        out : object
            Result of the evaluation.pandas.DataFrame
        """
        def n():
            """
            Return number of rows in groups

            This function is part of the public API
            """
            return len(data)

        if isinstance(self.stmt, str):
            # Add function n() that computes the
            # size of the group data to the inner namespace.
            namespace = dict(data, n=n)
            # Avoid obvious keywords e.g if a column
            # is named class
            if self.stmt not in KEYWORDS:
                value = env.eval(
                    self.stmt,
                    source_name='Expression.evaluate',
                    inner_namespace=namespace)
            else:
                value = namespace[self.stmt]
        elif callable(self.stmt):
            value = self.stmt(data)
        else:
            value = self.stmt
        return value
