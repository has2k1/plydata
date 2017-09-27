"""
Verb Initializations
"""
import itertools

from .operators import DataOperator, DoubleDataOperator

__all__ = ['define', 'create', 'sample_n', 'sample_frac', 'select',
           'rename', 'distinct', 'unique', 'arrange', 'group_by',
           'ungroup', 'group_indices', 'summarize', 'summarise',
           'query', 'do', 'head', 'tail', 'tally', 'count',
           'modify_where', 'define_where', 'mutate', 'transmute',
           'dropna', 'fillna', 'call',
           'inner_join', 'outer_join', 'left_join', 'right_join',
           'full_join', 'anti_join', 'semi_join']


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

    Note
    ----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`define` will modify the original dataframe.
    """
    new_columns = None
    expressions = None  # Expressions to create the new columns

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
        self.new_columns = list(itertools.chain(cols, kwargs.keys()))
        self.expressions = list(itertools.chain(exprs, kwargs.values()))


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
    args : tuple, optional
        Names of columns in dataframe. Normally, they are strings.
    startswith : str, optional
        All column names that start with this string will be included.
    endswith : str, optional
        All column names that end with this string will be included.
    contains : str, optional
        All column names that contain with this string will be included.
    matches : str or regex, optional
        All column names that match the string or a compiled regex pattern
        will be included.
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
       tail  whistle
    0     1        1
    1     2        2
    2     3        3
    >>> df >> select('whistle',  endswith='ail')
       nail  tail  whistle
    0     1     1        1
    1     2     2        2
    2     3     3        3
    >>> df >> select('bell',  matches='\w+tle$')
       bell  whistle
    0     1        1
    1     2        2
    2     3        3
    """
    def __init__(self, *args, startswith=None, endswith=None,
                 contains=None, matches=None, drop=False):
        self.args = args
        self.kwargs = dict(
            startswith=startswith, endswith=endswith,
            contains=contains, matches=matches, drop=drop)


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
       gong  pin  tail  whistle
    0     1    1     1        1
    1     2    2     2        2
    2     3    3     3        3
    >>> df >> rename({'flap': 'tail'}, pin='nail')
       bell  pin  flap  whistle
    0     1    1     1        1
    1     2    2     2        2
    2     3    3     3        3

    Note
    ----
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
    new_columns = None
    expressions = None  # Expressions to create the new columns

    def __init__(self, *args, **kwargs):
        self.set_env_from_verb_init()
        if len(args) == 1:
            if isinstance(args[0], str):
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

            self.new_columns = list(kwargs.keys())
            self.expressions = list(kwargs.values())
            self.columns.extend(self.new_columns)
        else:
            self.new_columns = []
            self.expressions = []


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
        self.expressions = [x for x in args]


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
       y  y-1  xplus1
    0  1    0       2
    1  2    1       6
    2  3    2       3
    3  4    3       3
    4  5    4       5
    5  6    5       1
    6  5    4       5

    Note
    ----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`group_by` will modify the original dataframe.
    """
    groups = None

    def __init__(self, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(*args, **kwargs)
        self.new_columns = list(self.new_columns)
        self.groups = list(self.new_columns)


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


class summarize(DataOperator):
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
    - ``{n}`` - Number of elements in current group. A special
      function is created and substituted in place of ``{n}``.

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

    >>> df >> group_by('y') >> summarize(y_count='{n}')
       y  y_count
    0  0        2
    1  1        2
    2  2        1
    3  3        1

    You can use ``{n}`` even when there are no groups.

    >>> df >> summarize('{n}')
       {n}
    0    6
    """
    new_columns = None
    epressions = None  # Expressions to create the summary columns

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

        self.new_columns = list(itertools.chain(cols, kwargs.keys()))
        self.expressions = list(itertools.chain(exprs, kwargs.values()))


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

    For more information see :meth:`pandas.DataFrame.query`. To query
    rows and columns with ``NaN`` values, use :class:`dropna`
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
    args : function, optional
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
    ...     (m, c), _, _, _ = np.linalg.lstsq(X, gdf.y)
    ...     return pd.DataFrame({'slope': [m], 'intercept': c})

    Define functions that take x and y values and compute the
    intercept and slope.

    >>> def slope(x, y):
    ...     return np.diff(y)[0] / np.diff(x)[0]
    ...
    >>> def intercept(x, y):
    ...     return y.values[0] - slope(x, y) * x.values[0]

    Demonstrating do

    >>> df >> group_by('z') >> do(least_squares)
       z  intercept  slope
    0  a        1.0    1.0
    1  b        6.0   -1.0

    We can get the same result, by passing separate functions
    that calculate the columns independently.

    >>> df2 = df >> group_by('z') >> do(
    ...     slope=lambda gdf: slope(gdf.x, gdf.y),
    ...     intercept=lambda gdf: intercept(gdf.x, gdf.y))
    >>> df2[['z', 'intercept', 'slope']]  # Ordered the same as above
       z  intercept  slope
    0  a        1.0    1.0
    1  b        6.0   -1.0

    The functions need not return numerical values. Pandas columns can
    hold any type of object. You could store result objects from more
    complicated models. Each model would be linked to a group. Notice
    that the group columns (``z`` in the above cases) are included in
    the result.

    Note
    ----
    You cannot have both a position argument and keyword
    arguments.
    """
    single_function = None
    columns = None
    functions = None

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "Unexpected positional and keyword arguments.")

        if len(args) > 1:
            raise ValueError(
                "Got more than one positional argument.")

        if args:
            self.single_function = args[0]
        else:
            self.columns = list(kwargs.keys())
            self.functions = list(kwargs.values())


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


class tally(DataOperator):
    """
    Tally observations by group

    ``tally`` is a convenient wrapper for summarise that will
    either call ``n`` or ``sum(n)`` depending on whether you're
    tallying for the first time, or re-tallying.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    weights : str or array-like, optional
        Weight of each row in the group.
    sort : bool, optional
        If ``True``, sort the resulting data in descending
        order.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': ['a', 'b', 'a', 'b', 'a', 'b'],
    ...     'w': [1, 2, 1, 2, 1, 2]})

    Without groups it is one large group

    >>> df >> tally()
       n
    0  6

    Sum of the weights

    >>> df >> tally('w')
        n
    0   9

    With groups

    >>> df >> group_by('y') >> tally()
       y  n
    0  a  3
    1  b  3

    With groups and weights

    >>> df >> group_by('y') >> tally('w')
       y  n
    0  a  3
    1  b  6

    Applying the weights to a column

    >>> df >> group_by('y') >> tally('x*w')
       y  n
    0  a  9
    1  b 24

    You can do that with :class:`summarize`

    >>> df >> group_by('y') >> summarize(n='sum(x*w)')
       y  n
    0  a  9
    1  b 24
    """

    def __init__(self, weights=None, sort=False):
        self.set_env_from_verb_init()
        self.weights = weights
        self.sort = sort


class count(DataOperator):
    """
    Count observations by group

    ``count`` is a convenient wrapper for summarise that will
    either call n or sum(n) depending on whether you’re
    tallying for the first time, or re-tallying. Similar to
    :class:`tally`, but it does the :class:`group_by` for you.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    *args : str, list
        Columns to group by.
    weights : str or array-like, optional
        Weight of each row in the group.
    sort : bool, optional
        If ``True``, sort the resulting data in descending
        order.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': ['a', 'b', 'a', 'b', 'a', 'b'],
    ...     'w': [1, 2, 1, 2, 1, 2]})

    Without groups it is one large group

    >>> df >> count()
       n
    0  6

    Sum of the weights

    >>> df >> count(weights='w')
        n
    0   9

    With groups

    >>> df >> count('y')
       y  n
    0  a  3
    1  b  3

    With groups and weights

    >>> df >> count('y', weights='w')
       y  n
    0  a  3
    1  b  6

    Applying the weights to a column

    >>> df >> count('y', weights='x*w')
       y  n
    0  a  9
    1  b 24

    You can do that with :class:`summarize`

    >>> df >> group_by('y') >> summarize(n='sum(x*w)')
       y  n
    0  a  9
    1  b 24
    """

    def __init__(self, *args, weights=None, sort=False):
        self.set_env_from_verb_init()
        if len(args) == 1:
            self.groups = [args[0]]
        elif len(args) > 1:
            self.groups = list(args)
        else:
            self.groups = []
        self.weights = weights
        self.sort = sort


class modify_where(DataOperator):
    """
    Modify columns from of selected rows

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    where : str
        The query to evaluate and find the rows to be modified.
        You can refer to variables in the environment by prefixing
        them with an '@' character like ``@a + b``. Allowed functions
        are `sin`, `cos`, `exp`, `log`, `expm1`, `log1p`, `sqrt`,
        `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`,
        `arccosh`, `arcsinh`, `arctanh`, `abs` and `arctan2`.
    args : tuple, optional
        A single positional argument that holds
        ``('column', expression)`` pairs. This is useful if
        the *column* is not a valid python variable name.
    kwargs : dict, optional
        ``{column: expression}`` pairs. If all the columns to
        be adjusted are valid python variable names, then they
        can be specified as keyword arguments.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'x': [0, 1, 2, 3, 4, 5],
    ...     'y': [0, 1, 2, 3, 4, 5],
    ...     'z': [0, 1, 2, 3, 4, 5]
    ... })
    >>> df >> modify_where('x%2 == 0', y='y*10', z='x-y')
       x   y  z
    0  0   0  0
    1  1   1  1
    2  2  20  0
    3  3   3  3
    4  4  40  0
    5  5   5  5

    Compared that to::

        idx = df['x'] % 2 == 0
        df.loc[idx, 'z'] = df.loc[idx, 'x'] - df.loc[idx, 'y']
        df.loc[idx, 'y'] = df.loc[idx, 'y'] * 10

    Note
    ----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`modify_where` will modify the original dataframe.

    The ``where`` query expression and the expressions for the
    ``args`` and ``kwargs`` use different evaluation engines.
    In ``where``, you cannot use any other function calls or
    refer to variables in the namespace without the ``@``
    symbol.

    To modify cells with ``NaN`` values use :class:`fillna`.
    """

    def __init__(self, where, *args, **kwargs):
        self.set_env_from_verb_init()
        self.where = where
        cols = []
        exprs = []
        for arg in args:
            try:
                col, expr = arg
            except (TypeError, ValueError):
                raise ValueError(
                    "Positional arguments must be a tuple of 2")
            cols.append(col)
            exprs.append(expr)

        self.columns = list(itertools.chain(cols, kwargs.keys()))
        self.expressions = list(itertools.chain(exprs, kwargs.values()))


class define_where(DataOperator):
    """
    Add column to DataFrame where the value is based on a condition

    This verb is a combination of :class:`define` and
    :class:`modify_where`.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    where : str
        The query to evaluate and find the rows to be modified.
        You can refer to variables in the environment by prefixing
        them with an '@' character like ``@a + b``. Allowed functions
        are `sin`, `cos`, `exp`, `log`, `expm1`, `log1p`, `sqrt`,
        `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`,
        `arccosh`, `arcsinh`, `arctanh`, `abs` and `arctan2`.
    args : tuple, optional
        A single positional argument that holds
        ``('column', 2-expressions)`` pairs. This is useful if
        the *column* is not a valid python variable name.
    kwargs : dict, optional
        ``{column: 2-expressions}`` pairs. If all the columns to
        be adjusted are valid python variable names, then they
        can be specified as keyword arguments.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5]})
    >>> df >> define_where('x%2 == 0', parity=("'even'", "'odd'"))
       x parity
    0  0   even
    1  1    odd
    2  2   even
    3  3    odd
    4  4   even
    5  5    odd

    This is equivalent to

    >>> (df
    ...  >> define(parity="'odd'")
    ...  >> modify_where('x%2 == 0', parity="'even'"))
       x parity
    0  0   even
    1  1    odd
    2  2   even
    3  3    odd
    4  4   even
    5  5    odd

    Note
    ----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`define_where` will modify the original dataframe.
    """
    new_columns = None
    expressions = None  # Expressions to create the new columns

    def __init__(self, where, *args, **kwargs):
        self.set_env_from_verb_init()
        self.where = where
        cols = []
        exprs = []
        for arg in args:
            if isinstance(arg, str):
                col = expr = arg
            else:
                col, expr = arg
            cols.append(col)
            exprs.append(expr)
        self.new_columns = list(itertools.chain(cols, kwargs.keys()))
        self.expressions = list(itertools.chain(exprs, kwargs.values()))


class dropna(DataOperator):
    """
    Remove rows or columns with missing values

    This is a wrapper around :meth:`pandas.DataFrame.dropna`. It
    is useful because you cannot :class:`query` ``NaN`` values.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, or tuple/list thereof
        Pass tuple or list to drop on multiple axes
    how : {'any', 'all'}
        * any : if any NA values are present, drop that label
        * all : if all values are NA, drop that label
    thresh : int, default None
        int value : require that many non-NA values
    subset : array-like
        Labels along other axis to consider, e.g. if you are
        dropping rows these would be a list of columns to include

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'w': [1, 2, np.nan, 4, 5],
    ...     'x': [np.nan, 2, np.nan, 4, 5],
    ...     'y': [np.nan] * 4 + [5],
    ...     'z': [np.nan] * 5
    ... })
    >>> df
         w    x    y   z
    0  1.0  NaN  NaN NaN
    1  2.0  2.0  NaN NaN
    2  NaN  NaN  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Drop rows with any ``NaN`` values

    >>> df >> dropna()
    Empty DataFrame
    Columns: [w, x, y, z]
    Index: []

    Drop rows with all ``NaN`` values

    >>> df >> dropna(how='all')
         w    x    y   z
    0  1.0  NaN  NaN NaN
    1  2.0  2.0  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Drop rows with ``NaN`` values in the *x* column.

    >>> df >> dropna(subset=['x'])
         w    x    y   z
    1  2.0  2.0  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Drop and keep rows atleast 3 ``non-NaN`` values

    >>> df >> dropna(thresh=3)
         w    x    y   z
    4  5.0  5.0  5.0 NaN

    Drop columns with all ``NaN`` values

    >>> df >> dropna(axis=1, how='all')
         w    x    y
    0  1.0  NaN  NaN
    1  2.0  2.0  NaN
    2  NaN  NaN  NaN
    3  4.0  4.0  NaN
    4  5.0  5.0  5.0

    Drop columns with any ``NaN`` values in row 3.

    >>> df >> dropna(axis=1, subset=[3])
         w    x
    0  1.0  NaN
    1  2.0  2.0
    2  NaN  NaN
    3  4.0  4.0
    4  5.0  5.0
    """

    def __init__(self, axis=0, how='any', thresh=None, subset=None):
        self.axis = axis
        self.how = how
        self.thresh = thresh
        self.subset = subset


class fillna(DataOperator):
    """
    Fill NA/NaN values using the specified method

    This is a wrapper around :meth:`pandas.DataFrame.fillna`. It
    is useful because you cannot :class:`modify_where` ``NaN``
    values.

    Parameters
    ----------
    value : scalar, dict, Series, or DataFrame
        Value to use to fill holes (e.g. 0), alternately a
        dict/Series/DataFrame of values specifying which value to
        use for each index (for a Series) or column (for a DataFrame).
        (values not in the dict/Series/DataFrame will not be filled).
        This value cannot be a list.
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series
        pad / ffill: propagate last valid observation forward to next
        valid backfill / bfill: use NEXT valid observation to fill gap
    axis : {0 or 'index', 1 or 'columns'}
    inplace : boolean, default False
        If True, fill in place. Note: this will modify any
        other views on this object, (e.g. a no-copy slice for a column
        in a DataFrame).
    limit : int, default None
        If method is specified, this is the maximum number of
        consecutive NaN values to forward/backward fill. In other
        words, if there is a gap with more than this number of
        consecutive NaNs, it will only be partially filled. If method
        is not specified, this is the maximum number of entries along
        the entire axis where NaNs will be filled. Must be greater
        than 0 if not None.
    downcast : dict, default is None
        a dict of item->dtype of what to downcast if possible, or the
        string 'infer' which will try to downcast to an appropriate
        equal type (e.g. float64 to int64 if possible)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'w': [1, 2, np.nan, 4, 5],
    ...     'x': [np.nan, 2, np.nan, 4, 5],
    ...     'y': [np.nan] * 4 + [5],
    ...     'z': [np.nan] * 5
    ... })
    >>> df
         w    x    y   z
    0  1.0  NaN  NaN NaN
    1  2.0  2.0  NaN NaN
    2  NaN  NaN  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Replace all ``NaN`` values with -1.

    >>> df >> fillna(-1)
         w    x    y    z
    0  1.0 -1.0 -1.0 -1.0
    1  2.0  2.0 -1.0 -1.0
    2 -1.0 -1.0 -1.0 -1.0
    3  4.0  4.0 -1.0 -1.0
    4  5.0  5.0  5.0 -1.0

    Replace all ``NaN`` values with the first ``non-NaN`` value *above
    in column*

    >>> df >> fillna(method='ffill')
         w    x    y   z
    0  1.0  NaN  NaN NaN
    1  2.0  2.0  NaN NaN
    2  2.0  2.0  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Replace all ``NaN`` values with the first ``non-NaN`` value *below
    in column*

    >>> df >> fillna(method='bfill')
         w    x    y   z
    0  1.0  2.0  5.0 NaN
    1  2.0  2.0  5.0 NaN
    2  4.0  4.0  5.0 NaN
    3  4.0  4.0  5.0 NaN
    4  5.0  5.0  5.0 NaN

    Replace atmost 2 ``NaN`` values with the first ``non-NaN`` value
    *below in column*

    >>> df >> fillna(method='bfill', limit=2)
         w    x    y   z
    0  1.0  2.0  NaN NaN
    1  2.0  2.0  NaN NaN
    2  4.0  4.0  5.0 NaN
    3  4.0  4.0  5.0 NaN
    4  5.0  5.0  5.0 NaN

    Replace all ``NaN`` values with the first ``non-NaN`` value to the
    *left in the row*

    >>> df >> fillna(method='ffill', axis=1)
         w    x    y    z
    0  1.0  1.0  1.0  1.0
    1  2.0  2.0  2.0  2.0
    2  NaN  NaN  NaN  NaN
    3  4.0  4.0  4.0  4.0
    4  5.0  5.0  5.0  5.0

    Replace all ``NaN`` values with the first ``non-NaN`` value to the
    *right in the row*

    >>> df >> fillna(method='bfill', axis=1)
         w    x    y   z
    0  1.0  NaN  NaN NaN
    1  2.0  2.0  NaN NaN
    2  NaN  NaN  NaN NaN
    3  4.0  4.0  NaN NaN
    4  5.0  5.0  5.0 NaN

    Note
    ----
    If :obj:`plydata.options.modify_input_data` is ``True``,
    :class:`modify_where` will modify the original dataframe.
    """

    def __init__(self, value=None, method=None, axis=None, limit=None,
                 downcast=None):
        self.value = value
        self.method = method
        self.axis = axis
        self.limit = limit
        self.downcast = downcast


class call(DataOperator):
    """
    Call external function or dataframe method

    This is a special verb; it turns regular functions and
    dataframe instance methods into verb instances that you
    can pipe to. It reduces the times one needs to break out
    of the piping workflow.

    Parameters
    ----------
    func : callable or str
        A function that accepts a dataframe as the first argument.
        Dataframe methods are specified using strings and
        preferrably they should start with a period,
        e.g ``'.reset_index'``
    *args : tuple
        Second, third, fourth, ... arguments to ``func``
    **kwargs : dict
        Keyword arguments to ``func``

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': {0: 'a', 1: 'b', 2: 'c'},
    ...     'B': {0: 1, 1: 3, 2: 5},
    ...     'C': {0: 2, 1: 4, 2: np.nan}
    ... })
    >>> df
       A  B    C
    0  a  1  2.0
    1  b  3  4.0
    2  c  5  NaN

    Using an external function

    >>> df >> call(pd.melt)
      variable value
    0        A     a
    1        A     b
    2        A     c
    3        B     1
    4        B     3
    5        B     5
    6        C     2
    7        C     4
    8        C   NaN

    An external function with arguments

    >>> df >> call(pd.melt, id_vars=['A'], value_vars=['B'])
       A variable  value
    0  a        B      1
    1  b        B      3
    2  c        B      5

    A method on the dataframe

    >>> df >> call('.dropna', axis=1)
       A  B
    0  a  1
    1  b  3
    2  c  5

    >>> (df
    ...  >> call(pd.melt)
    ...  >> query('variable != "B"')
    ...  >> call('.reset_index', drop=True)
    ...  )
      variable value
    0        A     a
    1        A     b
    2        A     c
    3        C     2
    4        C     4
    5        C   NaN
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

# Multiple Table Verbs


class _join(DoubleDataOperator):
    """
    Base class for join verbs
    """

    def __init__(self, x, y, on=None, suffixes=('_x', '_y')):
        self.x = x
        self.y = y
        self.kwargs = dict(on=on, suffixes=suffixes)


class inner_join(_join):
    """
    Join dataframes using the intersection of keys from both frames

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join
    suffixes : 2-length sequence
        Suffix to apply to overlapping column names in the left and
        right side, respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three'],
    ...     'col2': [1, 4, 3]
    ... })
    ...
    >>> inner_join(df1, df2, on='col1')
        col1  col2_x  col2_y
    0    one       1       1
    1  three       3       3

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """


class outer_join(_join):
    """
    Join dataframes using the union of keys from both frames

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join
    suffixes : 2-length sequence
        Suffix to apply to overlapping column names in the left and
        right side, respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three'],
    ...     'col2': [1, 4, 3]
    ... })
    ...
    >>> outer_join(df1, df2, on='col1')
        col1  col2_x  col2_y
    0    one     1.0     1.0
    1    two     2.0     NaN
    2  three     3.0     3.0
    3   four     NaN     4.0

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """


class left_join(_join):
    """
    Join dataframes using only keys from left frame

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join
    suffixes : 2-length sequence
        Suffix to apply to overlapping column names in the left and
        right side, respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three'],
    ...     'col2': [1, 4, 3]
    ... })
    ...
    >>> left_join(df1, df2, on='col1')
        col1  col2_x  col2_y
    0    one       1     1.0
    1    two       2     NaN
    2  three       3     3.0

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """


class right_join(_join):
    """
    Join dataframes using only keys from right frame

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join
    suffixes : 2-length sequence
        Suffix to apply to overlapping column names in the left and
        right side, respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three'],
    ...     'col2': [1, 4, 3]
    ... })
    ...
    >>> right_join(df1, df2, on='col1')
        col1  col2_x  col2_y
    0    one     1.0       1
    1  three     3.0       3
    2   four     NaN       4

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """


class anti_join(_join):
    """
    Join and keep rows only found in left frame

    Also keeps just the columns in the left frame. An ``anti_join``
    is analogous to a set difference.

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three'],
    ...     'col2': [1, 4, 3]
    ... })
    ...
    >>> anti_join(df1, df2, on='col1')
        col1  col2
    1    two     2

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """

    def __init__(self, x, y, on=None):
        self.x = x
        self.y = y
        self.kwargs = dict(on=on)


class semi_join(_join):
    """
    Join and keep columns only found in left frame & no duplicate rows

    A semi join differs from an inner join because an inner
    join will return one row of left frame for each matching row of
    the right, where a semi join will never duplicate rows of the
    left frame.

    Parameters
    ----------
    x : dataframe
        Left dataframe
    y : dataframe
        Right dataframe
    on : str or tuple or list
        Columns on which to join
    suffixes : 2-length sequence
        Suffix to apply to overlapping column names in the left and
        right side, respectively.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     'col1': ['one', 'two', 'three'],
    ...     'col2': [1, 2, 3]
    ... })
    ...
    >>> df2 = pd.DataFrame({
    ...     'col1': ['one', 'four', 'three', 'three'],
    ...     'col2': [1, 4, 3, 3]
    ... })
    ...
    >>> semi_join(df1, df2, on='col1')
        col1  col2
    0    one     1
    2  three     3

    Compared to an :class:`inner_join`

    >>> inner_join(df1, df2, on='col1')
        col1  col2_x  col2_y
    0    one       1       1
    1  three       3       3
    2  three       3       3

    Note
    ----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """


# Aliases
mutate = define
transmute = create
unique = distinct
summarise = summarize
full_join = outer_join
