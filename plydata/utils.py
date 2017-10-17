from contextlib import contextmanager

import pandas as pd

from .eval import EvalEnvironment


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
