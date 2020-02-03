"""
One table verb initializations
"""
import itertools

from .operators import DataOperator
from .expressions import Expression

__all__ = ['define', 'create', 'sample_n', 'sample_frac', 'select',
           'rename', 'distinct', 'unique', 'arrange', 'group_by',
           'ungroup', 'group_indices', 'summarize',
           'query', 'do', 'head', 'tail', 'pull', 'slice_rows',
           # Aliases
           'summarise', 'mutate', 'transmute',
           ]


class define(DataOperator):
    """
    Add column to DataFrame

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : strs, tuples, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe.
    kwargs : dict, optional
        ``{name: expression}`` pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3]})
    >>> df >> define(x_sq='x**2')
       x  x_sq
    0  1     1
    1  2     4
    2  3     9
    >>> df >> define(('x*2', 'x*2'), ('x*3', 'x*3'), x_cubed='x**3')
       x  x*2  x*3  x_cubed
    0  1    2    3        1
    1  2    4    6        8
    2  3    6    9       27
    >>> df >> define('x*4')
       x  x*4
    0  1    4
    1  2    8
    2  3   12

    Notes
    -----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`define` will modify the original dataframe.
    """

    def __init__(self, *args, **kwargs):
        self.set_env_from_verb_init()
        cols = []
        exprs = []
        for arg in args:
            if isinstance(arg, str):
                col = expr = arg
            else:
                col, expr = arg
            cols.append(col)
            exprs.append(expr)

        _cols = itertools.chain(cols, kwargs.keys())
        _exprs = itertools.chain(exprs, kwargs.values())
        self.expressions = [Expression(stmt, col)
                            for stmt, col in zip(_exprs, _cols)]


class create(define):
    """
    Create DataFrame with columns

    Similar to :class:`define`, but it drops the existing columns.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : strs, tuples, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe.
    kwargs : dict, optional
        ``{name: expression}`` pairs.

    kwargs : dict, optional
        ``{name: expression}`` pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3]})
    >>> df >> create(x_sq='x**2')
       x_sq
    0     1
    1     4
    2     9
    >>> df >> create(('x*2', 'x*2'), ('x*3', 'x*3'), x_cubed='x**3')
       x*2  x*3  x_cubed
    0    2    3        1
    1    4    6        8
    2    6    9       27
    >>> df >> create('x*4')
       x*4
    0    4
    1    8
    2   12
    """


class sample_n(DataOperator):
    """
    Sample n rows from dataframe

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    n : int, optional
        Number of items from axis to return.
    replace : boolean, optional
        Sample with or without replacement. Default = False.
    weights : str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        inf and -inf values not allowed.
    random_state : int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState
        object.
    axis : int or string, optional
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames, 1 for Panels).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rs = np.random.RandomState(1234567890)
    >>> df = pd.DataFrame({'x': range(20)})
    >>> df >> sample_n(5, random_state=rs)
         x
    5    5
    19  19
    14  14
    8    8
    17  17
    """
    def __init__(self, n=1, replace=False, weights=None,
                 random_state=None, axis=None):
        self.kwargs = dict(n=n, replace=replace, weights=weights,
                           random_state=random_state, axis=axis)


class sample_frac(DataOperator):
    """
    Sample a fraction of rows from dataframe

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.
    replace : boolean, optional
        Sample with or without replacement. Default = False.
    weights : str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        inf and -inf values not allowed.
    random_state : int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState
        object.
    axis : int or string, optional
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames, 1 for Panels).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rs = np.random.RandomState(1234567890)
    >>> df = pd.DataFrame({'x': range(20)})
    >>> df >> sample_frac(0.25, random_state=rs)
         x
    5    5
    19  19
    14  14
    8    8
    17  17
    """

    def __init__(self, frac=None, replace=False, weights=None,
                 random_state=None, axis=None):
        self.kwargs = dict(
            frac=frac, replace=replace, weights=weights,
            random_state=random_state, axis=axis)


class select(DataOperator):
    """
    Select columns by name

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple, optional
        Names of columns in dataframe. Normally, they are strings
        can include slice e.g :py:`slice('col2', 'col5')`.
        You can also exclude columns by prepending a ``-`` e.g
        py:`select('-col1')`, will include all columns minus than
        *col1*.
    startswith : str or tuple, optional
        All column names that start with this string will be included.
    endswith : str or tuple, optional
        All column names that end with this string will be included.
    contains : str or tuple, optional
        All column names that contain with this string will be included.
    matches : str or regex or tuple, optional
        All column names that match the string or a compiled regex pattern
        will be included. A tuple can be used to match multiple regexs.
    drop : bool, optional
        If ``True``, the selection is inverted. The unspecified/unmatched
        columns are returned instead. Default is ``False``.

    Examples
    --------
    >>> import pandas as pd
    >>> x = [1, 2, 3]
    >>> df = pd.DataFrame({'bell': x, 'whistle': x, 'nail': x, 'tail': x})
    >>> df >> select('bell', 'nail')
       bell  nail
    0     1     1
    1     2     2
    2     3     3
    >>> df >> select('bell', 'nail', drop=True)
       whistle  tail
    0        1     1
    1        2     2
    2        3     3
    >>> df >> select('whistle',  endswith='ail')
       whistle nail  tail
    0        1    1     1
    1        2    2     2
    2        3    3     3
    >>> df >> select('bell',  matches=r'\\w+tle$')
       bell  whistle
    0     1        1
    1     2        2
    2     3        3

    You can select column slices too. Like :meth:`~pandas.DataFrame.loc`,
    the stop column is included.

    >>> df = pd.DataFrame({'a': x, 'b': x, 'c': x, 'd': x,
    ...                    'e': x, 'f': x, 'g': x, 'h': x})
    >>> df
       a  b  c  d  e  f  g  h
    0  1  1  1  1  1  1  1  1
    1  2  2  2  2  2  2  2  2
    2  3  3  3  3  3  3  3  3
    >>> df >> select('a', slice('c', 'e'), 'g')
       a  c  d  e  g
    0  1  1  1  1  1
    1  2  2  2  2  2
    2  3  3  3  3  3

    You can exclude columns by prepending ``-``

    >>> df >> select('-a', '-c', '-e')
       b  d  f  g  h
    0  1  1  1  1  1
    1  2  2  2  2  2
    2  3  3  3  3  3

    Remove and place column at the end

    >>> df >> select('-a', '-c', '-e', 'a')
       b  d  f  g  h  a
    0  1  1  1  1  1  1
    1  2  2  2  2  2  2
    2  3  3  3  3  3  3

    Notes
    -----
    To exclude columns by prepending a minus, the first column
    passed to :class:`select` must be prepended with minus.
    :py:`select('-a', 'c')` will exclude column ``a``, while
    :py:`select('c', '-a')` will not exclude column ``a``.
    """
    def __init__(self, *names, startswith=None, endswith=None,
                 contains=None, matches=None, drop=False):
        def as_tuple(obj):
            if obj is None:
                return tuple()
            elif isinstance(obj, tuple):
                return obj
            elif isinstance(obj, list):
                return tuple(obj)
            else:
                return (obj,)

        self.names = names
        self.startswith = as_tuple(startswith)
        self.endswith = as_tuple(endswith)
        self.contains = as_tuple(contains)
        self.matches = as_tuple(matches)
        self.drop = drop

    @staticmethod
    def from_columns(*columns):
        """
        Create a select verb from the columns specification

        Parameters
        ----------
        *columns : list-like | select | str | slice
            Column names to be gathered and whose contents will
            make values.

        Return
        ------
        out : select
            Select verb representation of the columns.
        """
        from .helper_verbs import select_all, select_at, select_if
        n = len(columns)
        if n == 0:
            return select_all()
        elif n == 1:
            obj = columns[0]
            if isinstance(obj, (select, select_all, select_at, select_if)):
                return obj
            elif isinstance(obj, slice):
                return select(obj)
            elif isinstance(obj, (list, tuple)):
                return select(*obj)
            elif isinstance(obj, str):
                return select(obj)
            else:
                raise TypeError(
                    "Unrecognised type {}".format(type(obj))
                )
        else:
            return select(*columns)


class rename(DataOperator):
    """
    Rename columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : tuple, optional
        A single positional argument that holds
        ``{'new_name': 'old_name'}`` pairs. This is useful if the
        *old_name* is not a valid python variable name.
    kwargs : dict, optional
        ``{new_name: 'old_name'}`` pairs. If all the columns to be
        renamed are valid python variable names, then they
        can be specified as keyword arguments.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> df = pd.DataFrame({'bell': x, 'whistle': x,
    ...                    'nail': x, 'tail': x})
    >>> df >> rename(gong='bell', pin='nail')
       gong  whistle  pin  tail
    0     1        1    1     1
    1     2        2    2     2
    2     3        3    3     3
    >>> df >> rename({'flap': 'tail'}, pin='nail')
       bell  whistle  pin  flap
    0     1        1    1     1
    1     2        2    2     2
    2     3        3    3     3

    Notes
    -----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`rename` will modify the original dataframe.
    """
    lookup = None

    def __init__(self, *args, **kwargs):
        lookup = args[0] if len(args) else {}
        self.lookup = {v: k for k, v in lookup.items()}
        self.lookup.update({v: k for k, v in kwargs.items()})


class distinct(DataOperator):
    """
    Select distinct/unique rows

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    columns : list-like, optional
        Column names to use when determining uniqueness.
    keep : {'first', 'last', False}, optional
        - ``first`` : Keep the first occurence.
        - ``last`` : Keep the last occurence.
        - False : Do not keep any of the duplicates.

        Default is False.
    kwargs : dict, optional
        ``{name: expression}`` computed columns. If specified,
        these are taken together with the columns when determining
        unique rows.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 1, 2, 3, 4, 4, 5],
    ...                    'y': [1, 2, 3, 4, 5, 5, 6]})
    >>> df >> distinct()
       x  y
    0  1  1
    1  1  2
    2  2  3
    3  3  4
    4  4  5
    6  5  6
    >>> df >> distinct(['x'])
       x  y
    0  1  1
    2  2  3
    3  3  4
    4  4  5
    6  5  6
    >>> df >> distinct(['x'], 'last')
       x  y
    1  1  2
    2  2  3
    3  3  4
    5  4  5
    6  5  6
    >>> df >> distinct(z='x%2')
       x  y  z
    0  1  1  1
    2  2  3  0
    >>> df >> distinct(['x'], z='x%2')
       x  y  z
    0  1  1  1
    2  2  3  0
    3  3  4  1
    4  4  5  0
    6  5  6  1
    >>> df >> define(z='x%2') >> distinct(['x', 'z'])
       x  y  z
    0  1  1  1
    2  2  3  0
    3  3  4  1
    4  4  5  0
    6  5  6  1
    """
    columns = None
    keep = 'first'

    def __init__(self, *args, **kwargs):
        self.set_env_from_verb_init()
        if len(args) == 1:
            if isinstance(args[0], (str, bool)):
                self.keep = args[0]
            else:
                self.columns = args[0]
        elif len(args) == 2:
            self.columns, self.keep = args
        elif len(args) > 2:
            raise Exception("Too many positional arguments.")

        # define
        if kwargs:
            if self.columns is None:
                self.columns = []
            elif not isinstance(self.columns, list):
                self.columns = list(self.columns)

            _cols = list(kwargs.keys())
            _exprs = list(kwargs.values())
            self.columns.extend(_cols)
        else:
            _cols = []
            _exprs = []

        self.expressions = [Expression(stmt, col)
                            for stmt, col in zip(_exprs, _cols)]


class arrange(DataOperator):
    """
    Sort rows by column variables

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : tuple
        Columns/expressions to sort by.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0],
    ...                    'y': [1, 2, 3, 4, 5, 6]})
    >>> df >> arrange('x')
       x  y
    5  0  6
    0  1  1
    2  2  3
    3  2  4
    4  4  5
    1  5  2
    >>> df >> arrange('x', '-y')
       x  y
    5  0  6
    0  1  1
    3  2  4
    2  2  3
    4  4  5
    1  5  2
    >>> df >> arrange('np.sin(y)')
       x  y
    4  4  5
    3  2  4
    5  0  6
    2  2  3
    0  1  1
    1  5  2
    """
    expressions = None

    def __init__(self, *args):
        self.set_env_from_verb_init()
        name_gen = ('col_{}'.format(x) for x in range(100))
        self.expressions = [
            Expression(stmt, col)
            for stmt, col in zip(args, name_gen)
        ]


class group_by(define):
    """
    Group dataframe by one or more columns/variables

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : strs, tuples, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe.
    add_ : bool, optional
        If True, add to existing groups. Default is to create
        new groups.
    kwargs : dict, optional
        ``{name: expression}`` pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
    ...                    'y': [1, 2, 3, 4, 5, 6, 5]})
    >>> df >> group_by('x')
    groups: ['x']
       x  y
    0  1  1
    1  5  2
    2  2  3
    3  2  4
    4  4  5
    5  0  6
    6  4  5

    Like :meth:`define`, :meth:`group_by` creates any
    missing columns.

    >>> df >> group_by('y-1', xplus1='x+1')
    groups: ['y-1', 'xplus1']
       x  y  y-1  xplus1
    0  1  1    0       2
    1  5  2    1       6
    2  2  3    2       3
    3  2  4    3       3
    4  4  5    4       5
    5  0  6    5       1
    6  4  5    4       5

    Columns that are grouped on remain in the dataframe after any
    verb operations that do not use the group information. For
    example:

    >>> df >> group_by('y-1', xplus1='x+1') >> select('y')
    groups: ['y-1', 'xplus1']
       y-1  xplus1  y
    0    0       2  1
    1    1       6  2
    2    2       3  3
    3    3       3  4
    4    4       5  5
    5    5       1  6
    6    4       5  5

    Notes
    -----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`group_by` will modify the original dataframe.
    """
    groups = None

    def __init__(self, *args, add_=False, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(*args, **kwargs)
        self.add_ = add_
        self.groups = [expr.column for expr in self.expressions]


class ungroup(DataOperator):
    """
    Remove the grouping variables for dataframe

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3],
    ...                    'y': [1, 2, 3]})
    >>> df >> group_by('x')
    groups: ['x']
       x  y
    0  1  1
    1  2  2
    2  3  3
    >>> df >> group_by('x') >> ungroup()
       x  y
    0  1  1
    1  2  2
    2  3  3
    """


class group_indices(group_by):
    """
    Generate a unique id for each group

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : strs, tuples, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe. As this verb returns an array, the tuples have
        no added benefit over strings.
    kwargs : dict, optional
        ``{name: expression}`` pairs. As this verb returns an
        array, keyword arguments have no added benefit over
        :class:`str` positional arguments.

    Returns
    -------
    out : numpy.array
        Ids for each group

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
    ...                    'y': [1, 2, 3, 4, 5, 6, 5]})
    >>> df >> group_by('x')
    groups: ['x']
       x  y
    0  1  1
    1  5  2
    2  2  3
    3  2  4
    4  4  5
    5  0  6
    6  4  5
    >>> df >> group_by('x') >> group_indices()
    array([1, 4, 2, 2, 3, 0, 3])

    You can pass the group column(s) as parameters to
    :class:`group_indices`

    >>> df >> group_indices('x*2')
    array([1, 4, 2, 2, 3, 0, 3])
    """


class summarize(define):
    """
    Summarise multiple values to a single value

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    args : strs, tuples, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe.
    kwargs : dict, optional
        ``{name: expression}`` pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
    ...                    'y': [1, 2, 3, 4, 5, 6, 5],
    ...                    'z': [1, 3, 3, 4, 5, 5, 5]})

    Can take only positional, only keyword arguments or both.

    >>> df >> summarize('np.sum(x)', max='np.max(x)')
       np.sum(x)  max
    0         18    5

    When summarizing after a :class:`group_by` operation
    the group columns are retained.

    >>> df >> group_by('y', 'z') >> summarize(mean_x='np.mean(x)')
       y  z  mean_x
    0  1  1     1.0
    1  2  3     5.0
    2  3  3     2.0
    3  4  4     2.0
    4  5  5     4.0
    5  6  5     0.0

    .. rubric:: Aggregate Functions

    When summarizing the following functions can be used, they take
    an array and return a *single* number.

    - ``min(x)`` - Alias of :func:`numpy.amin` (a.k.a ``numpy.min``).
    - ``max(x)`` - Alias of :func:`numpy.amax` (a.k.a ``numpy.max``).
    - ``sum(x)`` - Alias of :func:`numpy.sum`.
    - ``cumsum(x)`` - Alias of :func:`numpy.cumsum`.
    - ``mean(x)`` - Alias of :func:`numpy.mean`.
    - ``median(x)`` - Alias of :func:`numpy.median`.
    - ``std(x)`` - Alias of :func:`numpy.std`.
    - ``first(x)`` - First element of ``x``.
    - ``last(x)`` - Last element of ``x``.
    - ``nth(x, n)`` - *nth* value of ``x`` or ``numpy.nan``.
    - ``n_distinct(x)`` - Number of distint elements in ``x``.
    - ``n_unique(x)`` - Alias of ``n_distinct``.
    - ``n()`` - Number of elements in current group.

    The aliases of the Numpy functions save you from typing 3 or 5 key
    strokes and you get better column names. i.e ``min(x)`` instead of
    ``np.min(x)`` or ``numpy.min(x)`` if you have Numpy imported.

    >>> df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
    ...                    'y': [0, 0, 1, 1, 2, 3]})
    >>> df >> summarize('min(x)', 'max(x)', 'mean(x)', 'sum(x)',
    ...                 'first(x)', 'last(x)', 'nth(x, 3)')
       min(x)  max(x)  mean(x)  sum(x)  first(x)  last(x)  nth(x, 3)
    0       0       5      2.5      15         0        5          3

    Summarizing groups with aggregate functions

    >>> df >> group_by('y') >> summarize('mean(x)')
       y  mean(x)
    0  0      0.5
    1  1      2.5
    2  2      4.0
    3  3      5.0

    >>> df >> group_by('y') >> summarize(y_count='n()')
       y  y_count
    0  0        2
    1  1        2
    2  2        1
    3  3        1

    You can use ``n()`` even when there are no groups.

    >>> df >> summarize('n()')
       n()
    0    6
    """


class query(DataOperator):
    """
    Return rows with matching conditions

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    expr : str
        The query string to evaluate.  You can refer to variables
        in the environment by prefixing them with an '@' character
        like ``@a + b``. Allowed functions are `sin`, `cos`, `exp`,
        `log`, `expm1`, `log1p`, `sqrt`, `sinh`, `cosh`, `tanh`,
        `arcsin`, `arccos`, `arctan`, `arccosh`, `arcsinh`,
        `arctanh`, `abs` and `arctan2`.
    kwargs : dict
        See the documentation for :func:`pandas.eval` for complete
        details on the keyword arguments accepted by
        :meth:`pandas.DataFrame.query`.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
    ...                    'y': [0, 0, 1, 1, 2, 3]})
    >>> df >> query('x % 2 == 0')
       x  y
    0  0  0
    2  2  1
    4  4  2

    >>> df >> query('x % 2 == 0 & y > 0')
       x  y
    2  2  1
    4  4  2

    By default, Bitwise operators ``&`` and ``|`` have the same
    precedence as the booleans ``and`` and ``or``.

    >>> df >> query('x % 2 == 0 and y > 0')
       x  y
    2  2  1
    4  4  2

    ``query`` works within groups

    >>> df >> query('x == x.min()')
       x  y
    0  0  0

    >>> df >> group_by('y') >> query('x == x.min()')
    groups: ['y']
       x  y
    0  0  0
    2  2  1
    4  4  2
    5  5  3

    For more information see :meth:`pandas.DataFrame.query`. To query
    rows and columns with ``NaN`` values, use :class:`dropna`

    Notes
    -----
    :class:`~plydata.one_table_verbs.query` is the equivalent of
    dplyr's `filter` verb but with slightly different python syntax
    the expressions.
    """
    expression = None

    def __init__(self, expr, **kwargs):
        self.set_env_from_verb_init()
        self.expression = expr
        self.kwargs = kwargs


class do(DataOperator):
    """
    Do arbitrary operations on a dataframe

    Considering the *split-apply-combine* data manipulation
    strategy, :class:`do` gives a window into which to place
    the complex *apply* actions, and also control over the form of
    results when they are combined. This allows

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    func : function, optional
        A single function to apply to each group. *The
        function should accept a dataframe and return a
        dataframe*.
    kwargs : dict, optional
        ``{name: function}`` pairs. *The function should
        accept a dataframe and return an array*. The function
        computes a column called ``name``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'x': [1, 2, 2, 3],
    ...                    'y': [2, 3, 4, 3],
    ...                    'z': list('aabb')})

    Define a function that uses numpy to do a least squares fit.
    It takes input from a dataframe and output is a dataframe.
    ``gdf`` is a dataframe that contains only rows from the current
    group.

    >>> def least_squares(gdf):
    ...     X = np.vstack([gdf.x, np.ones(len(gdf))]).T
    ...     (m, c), _, _, _ = np.linalg.lstsq(X, gdf.y, None)
    ...     return pd.DataFrame({'intercept': c, 'slope': [m]})

    Define functions that take x and y values and compute the
    intercept and slope.

    >>> def slope(x, y):
    ...     return np.diff(y)[0] / np.diff(x)[0]
    ...
    >>> def intercept(x, y):
    ...     return y.values[0] - slope(x, y) * x.values[0]

    Demonstrating do

    >>> df >> group_by('z') >> do(least_squares)
    groups: ['z']
       z  intercept  slope
    0  a        1.0    1.0
    1  b        6.0   -1.0

    We can get the same result, by passing separate functions
    that calculate the columns independently.

    >>> df >> group_by('z') >> do(
    ...     intercept=lambda gdf: intercept(gdf.x, gdf.y),
    ...     slope=lambda gdf: slope(gdf.x, gdf.y))
    groups: ['z']
       z  intercept  slope
    0  a        1.0    1.0
    1  b        6.0   -1.0

    The functions need not return numerical values. Pandas columns can
    hold any type of object. You could store result objects from more
    complicated models. Each model would be linked to a group. Notice
    that the group columns (``z`` in the above cases) are included in
    the result.

    Notes
    -----
    You cannot have both a position argument and keyword
    arguments.
    """
    single_function = False

    def __init__(self, func=None, **kwargs):
        if func is not None:
            if kwargs:
                raise ValueError(
                    "Unexpected positional and keyword arguments.")
            if not callable(func):
                raise TypeError(
                    "func should be a callable object")

        if func:
            self.single_function = True
            self.expressions = [Expression(func, None)]
        else:
            stmts_cols = zip(kwargs.values(), kwargs.keys())
            self.expressions = [
                Expression(stmt, col) for stmt, col in stmts_cols
            ]


class head(DataOperator):
    """
    Select the top n rows

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    n : int, optional
        Number of rows to return. If the ``data`` is grouped,
        then number of rows per group. Default is 5.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'y': list('aaaabbcddd') })
    >>> df >> head(2)
       x  y
    0  1  a
    1  2  a

    Grouped dataframe

    >>> df >> group_by('y') >> head(2)
    groups: ['y']
       x  y
    0  1  a
    1  2  a
    2  5  b
    3  6  b
    4  7  c
    5  8  d
    6  9  d
    """
    def __init__(self, n=5):
        self.n = n


class tail(DataOperator):
    """
    Select the bottom n rows

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    n : int, optional
        Number of rows to return. If the ``data`` is grouped,
        then number of rows per group. Default is 5.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'y': list('aaaabbcddd') })
    >>> df >> tail(2)
        x  y
    8   9  d
    9  10  d

    Grouped dataframe

    >>> df >> group_by('y') >> tail(2)
    groups: ['y']
        x  y
    0   3  a
    1   4  a
    2   5  b
    3   6  b
    4   7  c
    5   9  d
    6  10  d
    """
    def __init__(self, n=5):
        self.n = n


class pull(DataOperator):
    """
    Pull a single column from the dataframe

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    column : name
        Column name or index id.
    use_index : bool
        Whether to pull column by name or by its integer
        index. Default is False.


    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3],
    ...     'y': [4, 5, 6],
    ...     'z': [7, 8, 9]
    ... })
    >>> df
       x  y  z
    0  1  4  7
    1  2  5  8
    2  3  6  9
    >>> df >> pull('y')
    array([4, 5, 6])
    >>> df >> pull(0, True)
    array([1, 2, 3])
    >>> df >> pull(-1, True)
    array([7, 8, 9])

    Notes
    -----
    Always returns a numpy array.

    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`pull` will not make a copy the original column.
    """
    def __init__(self, column, use_index=False):
        self.column = column
        self.use_index = use_index


class slice_rows(DataOperator):
    """
    Select rows

    A wrapper around :class:`slice` to use when piping.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    *args : tuple
        (start, stop, step) as expected by the builtin :class:`slice`
        type.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': range(10), 'y': range(100, 110)})
    >>> df >> slice_rows(5)
       x    y
    0  0  100
    1  1  101
    2  2  102
    3  3  103
    4  4  104

    >>> df >> slice_rows(3, 7)
       x    y
    3  3  103
    4  4  104
    5  5  105
    6  6  106

    >>> df >> slice_rows(None, None, 3)
       x    y
    0  0  100
    3  3  103
    6  6  106
    9  9  109

    The above examples are equivalent to::

        df[slice(5)]
        df[slice(3, 7)]
        df[slice(None, None, 3)]

    respectively.

    Notes
    -----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`slice_rows` will not make a copy the original dataframe.
    """
    def __init__(self, *args):
        self.slice = slice(*args)


# Aliases
mutate = define
transmute = create
unique = distinct
summarise = summarize
