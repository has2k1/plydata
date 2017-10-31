"""
Verb Initializations
"""
import itertools

from .operators import DataOperator, DoubleDataOperator
from .utils import Expression

__all__ = ['define', 'create', 'sample_n', 'sample_frac', 'select',
           'rename', 'distinct', 'unique', 'arrange', 'group_by',
           'ungroup', 'group_indices', 'summarize',
           'query', 'do', 'head', 'tail', 'dropna', 'fillna', 'call',
           # Helpers
           'tally', 'count', 'add_tally', 'add_count',
           'arrange_all', 'arrange_at', 'arrange_if',
           'create_all', 'create_at', 'create_if',
           'create_all', 'create_at', 'create_if',
           'group_by_all', 'group_by_at', 'group_by_if',
           'mutate_all', 'mutate_at', 'mutate_if',
           'query_all', 'query_at', 'query_if',
           'rename_all', 'rename_at', 'rename_if',
           'select_all', 'select_at', 'select_if',
           'summarize_all', 'summarize_at', 'summarize_if',
           'modify_where', 'define_where', 'mutate', 'transmute',
           # Two table verbs
           'inner_join', 'outer_join', 'left_join', 'right_join',
           'full_join', 'anti_join', 'semi_join',
           # Aliases
           'summarise',
           'summarise_all', 'summarise_at', 'summarise_if',
           'transmute_all', 'transmute_at', 'transmute_if',
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

    Note
    ----
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
        Names of columns in dataframe. Normally, they are strings.
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
       tail  whistle
    0     1        1
    1     2        2
    2     3        3
    >>> df >> select('whistle',  endswith='ail')
       whistle nail  tail
    0        1    1     1
    1        2    2     2
    2        3    3     3
    >>> df >> select('bell',  matches='\w+tle$')
       bell  whistle
    0     1        1
    1     2        2
    2     3        3
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

    Note
    ----
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
    groups: ['z']
       z  intercept  slope
    0  a        1.0    1.0
    1  b        6.0   -1.0

    We can get the same result, by passing separate functions
    that calculate the columns independently.

    >>> df2 = df >> group_by('z') >> do(
    ...     slope=lambda gdf: slope(gdf.x, gdf.y),
    ...     intercept=lambda gdf: intercept(gdf.x, gdf.y))
    >>> df2[['z', 'intercept', 'slope']]  # Ordered the same as above
    groups: ['z']
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
    single_function = False

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "Unexpected positional and keyword arguments.")

        if len(args) > 1:
            raise ValueError(
                "Got more than one positional argument.")

        if args:
            self.single_function = True
            self.expressions = [Expression(args[0], None)]
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

        # self.columns = list(itertools.chain(cols, kwargs.keys()))
        # self.expressions = list(itertools.chain(exprs, kwargs.values()))
        _cols = itertools.chain(cols, kwargs.keys())
        _exprs = itertools.chain(exprs, kwargs.values())
        self.expressions = [Expression(stmt, col)
                            for stmt, col in zip(_exprs, _cols)]


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

        # Split up the expression into those that define the
        # column and those that modify
        _cols = list(itertools.chain(cols, kwargs.keys()))
        _exprs = itertools.chain(exprs, kwargs.values())
        _where = []
        _define = []
        for (stmt_where, stmt_define), col in zip(_exprs, _cols):
            _where.append(Expression(stmt_where, col))
            _define.append(Expression(stmt_define, col))

        self.define_expressions = _define
        self.where_expressions = _where


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


# Single table verb helpers


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


class add_tally(tally):
    """
    Add column with tally of items in each group

    Similar to :class:`tally`, but it adds a column and does
    not collapse the groups.

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

    >>> df >> add_tally()
       w  x  y  n
    0  1  1  a  6
    1  2  2  b  6
    2  1  3  a  6
    3  2  4  b  6
    4  1  5  a  6
    5  2  6  b  6

    Sum of the weights

    >>> df >> add_tally('w')
       w  x  y  n
    0  1  1  a  9
    1  2  2  b  9
    2  1  3  a  9
    3  2  4  b  9
    4  1  5  a  9
    5  2  6  b  9

    With groups

    >>> df >> group_by('y') >> add_tally()
    groups: ['y']
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  3
    2  1  3  a  3
    3  2  4  b  3
    4  1  5  a  3
    5  2  6  b  3

    With groups and weights

    >>> df >> group_by('y') >> add_tally('w')
    groups: ['y']
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  6
    2  1  3  a  3
    3  2  4  b  6
    4  1  5  a  3
    5  2  6  b  6

    Applying the weights to a column

    >>> df >> group_by('y') >> add_tally('x*w')
    groups: ['y']
       w  x  y   n
    0  1  1  a   9
    1  2  2  b  24
    2  1  3  a   9
    3  2  4  b  24
    4  1  5  a   9
    5  2  6  b  24

    Add tally is equivalent to using :func:`sum` or ``n()``
    in :class:`define`.

    >>> df >> group_by('y') >> define(n='sum(x*w)')
    groups: ['y']
       w  x  y   n
    0  1  1  a   9
    1  2  2  b  24
    2  1  3  a   9
    3  2  4  b  24
    4  1  5  a   9
    5  2  6  b  24

    >>> df >> group_by('y') >> define(n='n()')
    groups: ['y']
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  3
    2  1  3  a  3
    3  2  4  b  3
    4  1  5  a  3
    5  2  6  b  3

    Which is the same result as
    :py:`df >> group_by('y') >> add_tally()` above.

    See Also
    --------
    :class:`add_count`
    """


class add_count(count):
    """
    Add column with number of items in each group

    Similar to :class:`count`, but it adds a column and does
    not collapse the groups. It is also a shortcut of
    :class:`add_tally` that does the grouping.

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

    >>> df >> add_count()
       w  x  y  n
    0  1  1  a  6
    1  2  2  b  6
    2  1  3  a  6
    3  2  4  b  6
    4  1  5  a  6
    5  2  6  b  6

    Sum of the weights

    >>> df >> add_count(weights='w')
       w  x  y  n
    0  1  1  a  9
    1  2  2  b  9
    2  1  3  a  9
    3  2  4  b  9
    4  1  5  a  9
    5  2  6  b  9

    With groups

    >>> df >> add_count('y')
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  3
    2  1  3  a  3
    3  2  4  b  3
    4  1  5  a  3
    5  2  6  b  3

    >>> df >> group_by('y') >> add_count()
    groups: ['y']
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  3
    2  1  3  a  3
    3  2  4  b  3
    4  1  5  a  3
    5  2  6  b  3

    With groups and weights

    >>> df >> add_count('y', weights='w')
       w  x  y  n
    0  1  1  a  3
    1  2  2  b  6
    2  1  3  a  3
    3  2  4  b  6
    4  1  5  a  3
    5  2  6  b  6

    Applying the weights to a column

    >>> df >> add_count('y', weights='x*w')
       w  x  y   n
    0  1  1  a   9
    1  2  2  b  24
    2  1  3  a   9
    3  2  4  b  24
    4  1  5  a   9
    5  2  6  b  24

    You can do that with :class:`add_tally`

    >>> df >> group_by('y') >> add_tally('x*w') >> ungroup()
       w  x  y   n
    0  1  1  a   9
    1  2  2  b  24
    2  1  3  a   9
    3  2  4  b  24
    4  1  5  a   9
    5  2  6  b  24

    See Also
    --------
    :class:`add_tally`
    """


class _all(DataOperator):
    """
    Base class for *_all verbs

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.
    """
    selector = '_all'

    def __init__(self, functions, *args, **kwargs):
        if functions is None:
            functions = tuple()
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        self.set_env_from_verb_init()
        self.functions = functions
        self.args = args
        self.kwargs = kwargs


class _if(DataOperator):
    """
    Base class for *_if verbs

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.
    """
    selector = '_if'

    def __init__(self, predicate, functions=None, *args, **kwargs):
        if functions is None:
            functions = tuple()
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        self.set_env_from_verb_init()
        self.predicate = predicate
        self.functions = functions
        self.args = args
        self.kwargs = kwargs


class _at(select):
    """
    Base class for *_at verbs

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.
    """
    selector = '_at'

    def __init__(self, names, functions, *args, **kwargs):
        # Sort out the arguments to select
        if isinstance(names, (tuple, list)):
            args_select = names
            kwargs_select = {}
        elif isinstance(names, str):
            args_select = (names,)
            kwargs_select = {}
        elif isinstance(names, dict):
            args_select = tuple()
            kwargs_select = names
        else:
            raise TypeError(
                "Unexpected type for the names specification.")

        if functions is None:
            functions = tuple()
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        self.set_env_from_verb_init()
        super().__init__(*args_select, **kwargs_select)
        self.functions = functions
        self.args = args
        self.kwargs = kwargs


class arrange_all(_all):
    """
    Arrange by all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns before they are sorted:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

        Note that, the functions do not change the data, they only
        affect the sorting.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Arranging in ascending order.

    >>> df >> arrange_all()
      alpha beta theta  x  y   z
    1     a    a     d  2  5   9
    0     a    b     c  1  6   7
    2     a    b     e  3  4  11
    5     b    q     e  6  1  12
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10

    Arranging in descending order.

    >>> df >> arrange_all(pd.Series.rank, ascending=False)
      alpha beta theta  x  y   z
    4     b    u     d  5  2  10
    3     b    r     c  4  3   8
    5     b    q     e  6  1  12
    2     a    b     e  3  4  11
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9

    Note
    ----
    Do not use functions that change the order of the values in the
    array. Such functions are most likely the wrong candidates,
    they corrupt the data. Use function(s) that return values that
    can be sorted.
    """

    def __init__(self, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(functions, *args, **kwargs)


class arrange_if(_if):
    """
    Arrange by all column that match a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str
        Functions to alter the columns before they are sorted:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

        Note that, the functions do not change the data, they only
        affect the sorting.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Arranging by the columns with strings in ascending order.

    >>> df >> arrange_if('is_string')
      alpha beta theta  x  y   z
    1     a    a     d  2  5   9
    0     a    b     c  1  6   7
    2     a    b     e  3  4  11
    5     b    q     e  6  1  12
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10

    Arranging by the columns with strings in descending order.

    >>> df >> arrange_if('is_string', pd.Series.rank, ascending=False)
      alpha beta theta  x  y   z
    4     b    u     d  5  2  10
    3     b    r     c  4  3   8
    5     b    q     e  6  1  12
    2     a    b     e  3  4  11
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9

    It is easier to sort by only the numeric columns in descending
    order.

    >>> df >> arrange_if('is_numeric', np.negative)
      alpha beta theta  x  y   z
    5     b    q     e  6  1  12
    4     b    u     d  5  2  10
    3     b    r     c  4  3   8
    2     a    b     e  3  4  11
    1     a    a     d  2  5   9
    0     a    b     c  1  6   7

    Note
    ----
    Do not use functions that change the order of the values in the
    array. Such functions are most likely the wrong candidates,
    they corrupt the data. Use function(s) that return values that
    can be sorted.
    """


class arrange_at(_at):
    """
    Arrange by specific columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str, optional
        Functions to alter the columns before they are sorted:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

        Note that, the functions do not change the data, they only
        affect the sorting.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Arrange by explictily naming the columns to arrange by.
    This is not much different from :class:`arrange`.

    >>> df >> arrange_at(('alpha', 'z'))
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Arrange by dynamically selecting the columns to arrange
    by. Here we the selection is *beta* and *theta*.

    >>> df >> arrange_at(dict(contains='eta'))
      alpha beta theta  x  y   z
    1     a    a     d  2  5   9
    0     a    b     c  1  6   7
    2     a    b     e  3  4  11
    5     b    q     e  6  1  12
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10

    In descending order.

    >>> (df
    ...  >> arrange_at(
    ...     dict(contains='eta'),
    ...     pd.Series.rank, ascending=False)
    ... )
      alpha beta theta  x  y   z
    4     b    u     d  5  2  10
    3     b    r     c  4  3   8
    5     b    q     e  6  1  12
    2     a    b     e  3  4  11
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9

    Note
    ----
    Do not use functions that change the order of the values in the
    array. Such functions are most likely the wrong candidates,
    they corrupt the data. Use function(s) that return values that
    can be sorted.
    """
    def __init__(self, columns, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(columns, functions, *args, **kwargs)


class create_all(_all):
    """
    Create a new dataframe with all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Create a new dataframe by doubling the values of the input frame.

    >>> def double(s):
    ...     return s + s
    >>> df >> create_all(double)
      alpha beta theta   x   y   z
    0    aa   bb    cc   2  12  14
    1    aa   aa    dd   4  10  18
    2    aa   bb    ee   6   8  22
    3    bb   rr    cc   8   6  16
    4    bb   uu    dd  10   4  20
    5    bb   qq    ee  12   2  24

    Convert from centimetes to inches.

    >>> def inch(col, decimals=0):
    ...     return np.round(col/2.54, decimals)
    >>> def feet(col, decimals=0):
    ...     return np.round(col/30.48, decimals)
    >>> df >> select('x', 'y', 'z') >> create_all((inch, feet), decimals=2)
       x_inch  y_inch  z_inch  x_feet  y_feet  z_feet
    0    0.39    2.36    2.76    0.03    0.20    0.23
    1    0.79    1.97    3.54    0.07    0.16    0.30
    2    1.18    1.57    4.33    0.10    0.13    0.36
    3    1.57    1.18    3.15    0.13    0.10    0.26
    4    1.97    0.79    3.94    0.16    0.07    0.33
    5    2.36    0.39    4.72    0.20    0.03    0.39

    Group columns are always included, but they do not count towards
    the matched columns.

    >>> (df
    ...  >> select('x', 'y', 'z')
    ...  >> group_by('x')
    ...  >> create_all((inch, feet), decimals=2))
    groups: ['x']
       x  y_inch  z_inch  y_feet  z_feet
    0  1    2.36    2.76    0.20    0.23
    1  2    1.97    3.54    0.16    0.30
    2  3    1.57    4.33    0.13    0.36
    3  4    1.18    3.15    0.10    0.26
    4  5    0.79    3.94    0.07    0.33
    5  6    0.39    4.72    0.03    0.39
    """


class create_if(_if):
    """
    Create a new dataframe with columns selected by a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str, optional
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Create a new dataframe by doubling selected column values of the
    input frame. ``'is_integer'`` is a shortcut to
    :py:`pdtypes.is_integer_dtype`.

    >>> def double(s):
    ...     return s + s
    >>> df >> create_if('is_integer', double)
        x   y   z
    0   2  12  14
    1   4  10  18
    2   6   8  22
    3   8   6  16
    4  10   4  20
    5  12   2  24

    Convert from centimetes to inches.

    >>> def inch(col, decimals=0):
    ...     return np.round(col/2.54, decimals)
    >>> def feet(col, decimals=0):
    ...     return np.round(col/30.48, decimals)
    >>> df >> create_if('is_integer', (inch, feet), decimals=2)
       x_inch  y_inch  z_inch  x_feet  y_feet  z_feet
    0    0.39    2.36    2.76    0.03    0.20    0.23
    1    0.79    1.97    3.54    0.07    0.16    0.30
    2    1.18    1.57    4.33    0.10    0.13    0.36
    3    1.57    1.18    3.15    0.13    0.10    0.26
    4    1.97    0.79    3.94    0.16    0.07    0.33
    5    2.36    0.39    4.72    0.20    0.03    0.39

    Group columns are always included, but they do not count towards
    the matched columns.

    >>> (df
    ...  >> group_by('x')
    ...  >> create_if('is_integer', (inch, feet), decimals=2))
    groups: ['x']
       x  y_inch  z_inch  y_feet  z_feet
    0  1    2.36    2.76    0.20    0.23
    1  2    1.97    3.54    0.16    0.30
    2  3    1.57    4.33    0.13    0.36
    3  4    1.18    3.15    0.10    0.26
    4  5    0.79    3.94    0.07    0.33
    5  6    0.39    4.72    0.03    0.39

    Selecting columns that match a predicate.

    >>> df >> create_if('is_integer')
       x  y   z
    0  1  6   7
    1  2  5   9
    2  3  4  11
    3  4  3   8
    4  5  2  10
    5  6  1  12
    """


class create_at(_at):
    """
    Create dataframe with specific columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Create a new dataframe by doubling selected column values of the input
    frame.

    >>> def double(s):
    ...     return s + s
    >>> df >> create_at(('x', 'y', 'z'), double)
        x   y   z
    0   2  12  14
    1   4  10  18
    2   6   8  22
    3   8   6  16
    4  10   4  20
    5  12   2  24

    Convert from centimetes to inches.

    >>> def inch(col, decimals=0):
    ...     return np.round(col/2.54, decimals)
    >>> def feet(col, decimals=0):
    ...     return np.round(col/30.48, decimals)
    >>> df >> create_at(('x', 'y', 'z'), (inch, feet), decimals=2)
       x_inch  y_inch  z_inch  x_feet  y_feet  z_feet
    0    0.39    2.36    2.76    0.03    0.20    0.23
    1    0.79    1.97    3.54    0.07    0.16    0.30
    2    1.18    1.57    4.33    0.10    0.13    0.36
    3    1.57    1.18    3.15    0.13    0.10    0.26
    4    1.97    0.79    3.94    0.16    0.07    0.33
    5    2.36    0.39    4.72    0.20    0.03    0.39

    Group columns are always included and if listed in the
    selection, the functions act on them.

    >>> (df
    ...  >> group_by('x')
    ...  >> create_at(('x', 'y', 'z'), (inch, feet), decimals=2))
    groups: ['x']
       x  x_inch  y_inch  z_inch  x_feet  y_feet  z_feet
    0  1    0.39    2.36    2.76    0.03    0.20    0.23
    1  2    0.79    1.97    3.54    0.07    0.16    0.30
    2  3    1.18    1.57    4.33    0.10    0.13    0.36
    3  4    1.57    1.18    3.15    0.13    0.10    0.26
    4  5    1.97    0.79    3.94    0.16    0.07    0.33
    5  6    2.36    0.39    4.72    0.20    0.03    0.39

    Group columns that are not listed are not acted upon by the
    functions.

    >>> (df
    ...  >> group_by('x')
    ...  >> create_at(dict(matches=r'x|y|z'), (inch, feet), decimals=2))
    groups: ['x']
       x  y_inch  z_inch  y_feet  z_feet
    0  1    2.36    2.76    0.20    0.23
    1  2    1.97    3.54    0.16    0.30
    2  3    1.57    4.33    0.13    0.36
    3  4    1.18    3.15    0.10    0.26
    4  5    0.79    3.94    0.07    0.33
    5  6    0.39    4.72    0.03    0.39
    """


class group_by_all(_all):
    """
    Groupby all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Grouping by all the columns

    >>> df >> group_by_all()
    groups: ['alpha', 'beta', 'theta', 'x', 'y', 'z']
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Grouping by all columns created by a function.
    Same output as above, but now all the columns are
    categorical

    >>> result = df >> group_by_all(pd.Categorical)
    >>> result
    groups: ['alpha', 'beta', 'theta', 'x', 'y', 'z']
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12
    >>> result['x']
    0    1
    1    2
    2    3
    3    4
    4    5
    5    6
    Name: x, dtype: category
    Categories (6, int64): [1, 2, 3, 4, 5, 6]

    If apply more than one function or  provide a postfix,
    the original columns are retained.

    >>> (df
    ...  >> select('x', 'y', 'z')
    ...  >> group_by_all(dict(cat=pd.Categorical)))
    groups: ['x_cat', 'y_cat', 'z_cat']
       x  y   z x_cat y_cat z_cat
    0  1  6   7     1     6     7
    1  2  5   9     2     5     9
    2  3  4  11     3     4    11
    3  4  3   8     4     3     8
    4  5  2  10     5     2    10
    5  6  1  12     6     1    12
    """
    def __init__(self, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(functions, *args, **kwargs)


class group_by_if(_if):
    """
    Group by selected columns that are true for a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.

    functions : callable or tuple or dict or str, optional
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Group by all string type columns. ``'is_string'`` is a
    shortcut to :func:`pandas.api.types.is_string_dtype`.

    >>> df >> group_by_if('is_string')
    groups: ['alpha', 'beta', 'theta']
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Applying a function to create the group columns

    >>> def double(s):
    ...     return s + s
    >>> df >> group_by_if('is_string', double)
    groups: ['alpha', 'beta', 'theta']
      alpha beta theta  x  y   z
    0    aa   bb    cc  1  6   7
    1    aa   aa    dd  2  5   9
    2    aa   bb    ee  3  4  11
    3    bb   rr    cc  4  3   8
    4    bb   uu    dd  5  2  10
    5    bb   qq    ee  6  1  12

    Apply more than one function, increases the number of
    columns

    >>> def m10(x): return x-10  # minus
    >>> def p10(x): return x+10  # plus
    >>> df >> group_by_if('is_numeric', (m10, p10))
    groups: ['x_m10', 'y_m10', 'z_m10', 'x_p10', 'y_p10', 'z_p10']
      alpha beta theta  x  y   z  x_m10  y_m10  z_m10  x_p10  y_p10  z_p10
    0     a    b     c  1  6   7     -9     -4     -3     11     16     17
    1     a    a     d  2  5   9     -8     -5     -1     12     15     19
    2     a    b     e  3  4  11     -7     -6      1     13     14     21
    3     b    r     c  4  3   8     -6     -7     -2     14     13     18
    4     b    u     d  5  2  10     -5     -8      0     15     12     20
    5     b    q     e  6  1  12     -4     -9      2     16     11     22
    """


class group_by_at(_at):
    """
    Group by select columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str, optional
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    In the simplest form it is not too different from
    :class:`group_by`.

    >>> df >> group_by_at(('x', 'y'))
    groups: ['x', 'y']
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    The power comes from the ability to do dynamic column selection.
    For example, regex match column names and apply function to get the
    group columns.

    >>> def double(s): return s + s
    >>> df >> group_by_at(dict(matches=r'\w+eta$'), double)
    groups: ['beta', 'theta']
      alpha beta theta  x  y   z
    0     a   bb    cc  1  6   7
    1     a   aa    dd  2  5   9
    2     a   bb    ee  3  4  11
    3     b   rr    cc  4  3   8
    4     b   uu    dd  5  2  10
    5     b   qq    ee  6  1  12
    """
    def __init__(self, columns, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(columns, functions, *args, **kwargs)


class mutate_all(_all):
    """
    Modify all columns that are true for a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    A single function with an argument

    >>> df >> select('x', 'y', 'z') >> mutate_all(np.add, 10)
        x   y   z
    0  11  16  17
    1  12  15  19
    2  13  14  21
    3  14  13  18
    4  15  12  20
    5  16  11  22

    A two functions that accept the same argument

    >>> (df
    ...  >> select('x', 'z')
    ...  >> mutate_all((np.add, np.subtract), 10)
    ... )
       x   z  x_add  z_add  x_subtract  z_subtract
    0  1   7     11     17          -9          -3
    1  2   9     12     19          -8          -1
    2  3  11     13     21          -7           1
    3  4   8     14     18          -6          -2
    4  5  10     15     20          -5           0
    5  6  12     16     22          -4           2

    Convert *x*, *y* and *z* from centimeters to inches and
    round the 2 decimal places.

    >>> (df
    ...  >> select('x', 'y', 'z')
    ...  >> mutate_all(dict(inch=lambda col: np.round(col/2.54, 2)))
    ... )
       x  y   z  x_inch  y_inch  z_inch
    0  1  6   7    0.39    2.36    2.76
    1  2  5   9    0.79    1.97    3.54
    2  3  4  11    1.18    1.57    4.33
    3  4  3   8    1.57    1.18    3.15
    4  5  2  10    1.97    0.79    3.94
    5  6  1  12    2.36    0.39    4.72

    Groupwise standardization of multiple variables.

    >>> def scale(col): return (col - np.mean(col))/np.std(col)
    >>> (df
    ...  >> group_by('alpha')
    ...  >> select('x', 'y', 'z')
    ...  >> mutate_all(scale))
    groups: ['alpha']
      alpha         x         y         z
    0     a -1.224745  1.224745 -1.224745
    1     a  0.000000  0.000000  0.000000
    2     a  1.224745 -1.224745  1.224745
    3     b -1.224745  1.224745 -1.224745
    4     b  0.000000  0.000000  0.000000
    5     b  1.224745 -1.224745  1.224745
    """


class mutate_if(_if):
    """
    Modify selected columns that are true for a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandas.api.types as pdtypes
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    A single function with an argument

    >>> df >> mutate_if(pdtypes.is_numeric_dtype, np.add, 10)
      alpha beta theta   x   y   z
    0     a    b     c  11  16  17
    1     a    a     d  12  15  19
    2     a    b     e  13  14  21
    3     b    r     c  14  13  18
    4     b    u     d  15  12  20
    5     b    q     e  16  11  22

    A two functions that accept the same argument and using our
    crude column selector.

    >>> def is_x_or_z(col): return col.name in ('x', 'z')
    >>> df >> mutate_if(is_x_or_z, (np.add, np.subtract), 10)
      alpha beta theta  x  y   z  x_add  z_add  x_subtract  z_subtract
    0     a    b     c  1  6   7     11     17          -9          -3
    1     a    a     d  2  5   9     12     19          -8          -1
    2     a    b     e  3  4  11     13     21          -7           1
    3     b    r     c  4  3   8     14     18          -6          -2
    4     b    u     d  5  2  10     15     20          -5           0
    5     b    q     e  6  1  12     16     22          -4           2

    Convert *x*, *y* and *z* from centimeters to inches and
    round the 2 decimal places.

    >>> (df
    ...  >> mutate_if('is_numeric',
    ...               dict(inch=lambda col: np.round(col/2.54, 2))))
      alpha beta theta  x  y   z  x_inch  y_inch  z_inch
    0     a    b     c  1  6   7    0.39    2.36    2.76
    1     a    a     d  2  5   9    0.79    1.97    3.54
    2     a    b     e  3  4  11    1.18    1.57    4.33
    3     b    r     c  4  3   8    1.57    1.18    3.15
    4     b    u     d  5  2  10    1.97    0.79    3.94
    5     b    q     e  6  1  12    2.36    0.39    4.72

    Groupwise standardization of multiple variables.

    >>> def scale(col): return (col - np.mean(col))/np.std(col)
    >>> (df
    ...  >> group_by('alpha')
    ...  >> mutate_if('is_numeric', scale))
    groups: ['alpha']
      alpha beta theta         x         y         z
    0     a    b     c -1.224745  1.224745 -1.224745
    1     a    a     d  0.000000  0.000000  0.000000
    2     a    b     e  1.224745 -1.224745  1.224745
    3     b    r     c -1.224745  1.224745 -1.224745
    4     b    u     d  0.000000  0.000000  0.000000
    5     b    q     e  1.224745 -1.224745  1.224745

    Using a boolean array to select the columns.

    >>> df >> mutate_if(
    ...     [False, False, False, True, True, True],
    ...     np.negative)
      alpha beta theta  x  y   z
    0     a    b     c -1 -6  -7
    1     a    a     d -2 -5  -9
    2     a    b     e -3 -4 -11
    3     b    r     c -4 -3  -8
    4     b    u     d -5 -2 -10
    5     b    q     e -6 -1 -12
    """


class mutate_at(_at):
    """
    Change selected columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    A single function with an argument

    >>> df >> mutate_at(('x', 'y', 'z'), np.add, 10)
      alpha beta theta   x   y   z
    0     a    b     c  11  16  17
    1     a    a     d  12  15  19
    2     a    b     e  13  14  21
    3     b    r     c  14  13  18
    4     b    u     d  15  12  20
    5     b    q     e  16  11  22

    A two functions that accept the same argument

    >>> df >> mutate_at(('x', 'z'), (np.add, np.subtract), 10)
      alpha beta theta  x  y   z  x_add  z_add  x_subtract  z_subtract
    0     a    b     c  1  6   7     11     17          -9          -3
    1     a    a     d  2  5   9     12     19          -8          -1
    2     a    b     e  3  4  11     13     21          -7           1
    3     b    r     c  4  3   8     14     18          -6          -2
    4     b    u     d  5  2  10     15     20          -5           0
    5     b    q     e  6  1  12     16     22          -4           2

    Convert *x*, *y* and *z* from centimeters to inches and
    round the 2 decimal places.

    >>> (df
    ...  >> mutate_at(('x', 'y', 'z'),
    ...               dict(inch=lambda col: np.round(col/2.54, 2)))
    ... )
      alpha beta theta  x  y   z  x_inch  y_inch  z_inch
    0     a    b     c  1  6   7    0.39    2.36    2.76
    1     a    a     d  2  5   9    0.79    1.97    3.54
    2     a    b     e  3  4  11    1.18    1.57    4.33
    3     b    r     c  4  3   8    1.57    1.18    3.15
    4     b    u     d  5  2  10    1.97    0.79    3.94
    5     b    q     e  6  1  12    2.36    0.39    4.72

    Groupwise standardization of multiple variables.

    >>> def scale(col): return (col - np.mean(col))/np.std(col)
    >>> (df
    ...  >> group_by('alpha')
    ...  >> mutate_at(('x', 'y', 'z'), scale))
    groups: ['alpha']
      alpha beta theta         x         y         z
    0     a    b     c -1.224745  1.224745 -1.224745
    1     a    a     d  0.000000  0.000000  0.000000
    2     a    b     e  1.224745 -1.224745  1.224745
    3     b    r     c -1.224745  1.224745 -1.224745
    4     b    u     d  0.000000  0.000000  0.000000
    5     b    q     e  1.224745 -1.224745  1.224745
    """


class query_all(_all):
    """
    Query all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    all_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns, the
        *union* (``|``), is used to select the output rows.
    any_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns, the
        *intersection* (``&``), is used to select the output rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select all rows where any of the entries along the columns
    is a 4.

    >>> df >> query_all(any_vars='({_} == 4)')
      alpha beta theta  x  y   z
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8

    The opposit, select all rows where none of the entries along
    the columns is a 4.

    >>> df >> query_all(all_vars='({_} != 4)')
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    For something more complicated, group-wise selection.

    Select groups where any of the columns a large (> 28) sum.
    First by using :class:`summarize_all`, we see that there is
    one such group. Then using :class:`query_all` selects it.

    >>> (df
    ...  >> group_by('alpha')
    ...  >> select('x', 'y', 'z')
    ...  >> summarize_all('sum'))
      alpha   x   y   z
    0     a   6  15  27
    1     b  15   6  30
    >>> (df
    ...  >> group_by('alpha')
    ...  >> select('x', 'y', 'z')
    ...  >> query_all(any_vars='(sum({_}) > 28)'))
    groups: ['alpha']
      alpha  x  y   z
    3     b  4  3   8
    4     b  5  2  10
    5     b  6  1  12

    Note that ``sum({_}) > 28`` is a column operation, it returns
    a single number for the whole column. Therefore the whole column
    is either selected or not selected. Column operations are what
    enable group-wise selection.
    """
    vars_predicate = None

    def __init__(self, *, all_vars=None, any_vars=None):
        self.set_env_from_verb_init()
        if all_vars and any_vars:
            raise ValueError(
                "Only one of `all_vars` or `any_vars` should "
                "be given."
            )
        elif all_vars:
            self.vars_predicate = all_vars
            self.all_vars = True
            self.any_vars = False
        elif any_vars:
            self.vars_predicate = any_vars
            self.any_vars = True
            self.all_vars = False
        else:
            raise ValueError(
                "One of `all_vars` or `any_vars` should be given.")


class query_if(_if):
    """
    Query all columns that match a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    all_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns selected by the
        *predicate*, the *union* (``|``), is used to select the output rows.
    any_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns selected by the
        predicate, *intersection* (``&``), is used to select the output rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select all rows where any of the entries along the integer columns
    is a 4.

    >>> df >> query_if('is_integer', any_vars='({_} == 4)')
      alpha beta theta  x  y   z
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8

    The opposite, select all rows where none of the entries along
    the integer columns is a 4.

    >>> df >> query_if('is_integer', all_vars='({_} != 4)')
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    For something more complicated, group-wise selection.

    Select groups where any of the columns a large (> 28) sum.
    First by using :class:`summarize_if`, we see that there is
    one such group. Then using :class:`query_if` selects it.

    >>> (df
    ...  >> group_by('alpha')
    ...  >> summarize_if('is_integer', 'sum'))
      alpha   x   y   z
    0     a   6  15  27
    1     b  15   6  30
    >>> (df
    ...  >> group_by('alpha')
    ...  >> query_if('is_integer', any_vars='(sum({_}) > 28)'))
    groups: ['alpha']
      alpha beta theta  x  y   z
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Note that ``sum({_}) > 28`` is a column operation, it returns
    a single number for the whole column. Therefore the whole column
    is either selected or not selected. Column operations are what
    enable group-wise selection.
    """
    vars_predicate = None

    def __init__(self, predicate, *, all_vars=None, any_vars=None):
        self.set_env_from_verb_init()
        self.predicate = predicate

        if all_vars and any_vars:
            raise ValueError(
                "Only one of `all_vars` or `any_vars` should "
                "be given."
            )
        elif all_vars:
            self.vars_predicate = all_vars
            self.all_vars = True
            self.any_vars = False
        elif any_vars:
            self.vars_predicate = any_vars
            self.any_vars = True
            self.all_vars = False
        else:
            raise ValueError(
                "One of `all_vars` or `any_vars` should be given.")


class query_at(_at):
    """
    Query specific columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    all_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns selected by the
        *names* specification, the *union* (``|``), is used to select the
        output rows.
    any_vars : str, optional
        A predicate statement to evaluate. It should conform to python
        syntax and should return an array of boolean values (one for every
        item in the column) or a single boolean (for the whole column).
        You should use ``{_}`` to refer to the column names.

        After the statement is evaluated for all columns selected by the
        *names* specification, *intersection* (``&``), is used to select
        the output rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select all rows where any of the entries along the integer columns
    is a 4.

    >>> df >> query_at(('x', 'y', 'z'), any_vars='({_} == 4)')
      alpha beta theta  x  y   z
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8

    The opposit, select all rows where none of the entries along
    the integer columns is a 4.

    >>> df >> query_at(('x', 'y', 'z'), all_vars='({_} != 4)')
      alpha beta theta  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    For something more complicated, group-wise selection.

    Select groups where any of the columns a large (> 28) sum.
    First by using :class:`summarize_at`, we see that there is
    one such group. Then using :class:`query_at` selects it.

    >>> (df
    ...  >> group_by('alpha')
    ...  >> summarize_at(('x', 'y', 'z'), 'sum'))
      alpha   x   y   z
    0     a   6  15  27
    1     b  15   6  30
    >>> (df
    ...  >> group_by('alpha')
    ...  >> query_at(('x', 'y', 'z'), any_vars='(sum({_}) > 28)'))
    groups: ['alpha']
      alpha beta theta  x  y   z
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Note that ``sum({_}) > 28`` is a column operation, it returns
    a single number for the whole column. Therefore the whole column
    is either selected or not selected. Column operations are what
    enable group-wise selection.
    """

    def __init__(self, names, *, all_vars=None, any_vars=None):
        if all_vars and any_vars:
            raise ValueError(
                "Only one of `all_vars` or `any_vars` should "
                "be given."
            )
        elif all_vars:
            self.vars_predicate = all_vars
            self.all_vars = True
            self.any_vars = False
        elif any_vars:
            self.vars_predicate = any_vars
            self.any_vars = True
            self.all_vars = False
        else:
            raise ValueError(
                "One of `all_vars` or `any_vars` should be given.")

        self.set_env_from_verb_init()
        super().__init__(names, tuple())


class rename_all(_all):
    """
    Rename all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Rename all columns uppercase

    >>> df >> rename_all(str.upper)
      ALPHA BETA THETA  X  Y   Z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Group columns are not renamed

    >>> df >> group_by('beta') >> rename_all(str.upper)
    groups: ['beta']
      ALPHA beta THETA  X  Y   Z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12
    """


class rename_if(_if):
    """
    Rename all columns that match a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments
        are passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    >>> def is_sorted(col):
    ...     a = col.values
    ...     return all(a[:-1] <= a[1:])

    Rename all sorted columns to uppercase.

    >>> df >> rename_if(is_sorted, str.upper)
      ALPHA beta theta  X  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Group columns are not renamed.

    >>> df >> group_by('alpha') >> rename_if(is_sorted, str.upper)
    groups: ['alpha']
      alpha beta theta  X  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12
    """


class rename_at(_at):
    """
    Rename specific columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Rename columns that contain the string ``eta`` to upper case.

    >>> df >> rename_at(dict(contains='eta'), str.upper)
      alpha BETA THETA  x  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    The group columns are not renamed.

    >>> (df
    ...  >> group_by('beta')
    ...  >> rename_at(('alpha', 'beta', 'x'), str.upper))
    groups: ['beta']
      ALPHA beta theta  X  y   z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12
    """


class select_all(_all):
    """
    Select all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select all columns and convert names to uppercase

    >>> df >> select_all(str.upper)
      ALPHA BETA THETA  X  Y   Z
    0     a    b     c  1  6   7
    1     a    a     d  2  5   9
    2     a    b     e  3  4  11
    3     b    r     c  4  3   8
    4     b    u     d  5  2  10
    5     b    q     e  6  1  12

    Group columns are selected but they are not renamed.

    >>> df >> group_by('beta') >> select_all(str.upper)
    groups: ['beta']
      beta ALPHA THETA  X  Y   Z
    0    b     a     c  1  6   7
    1    a     a     d  2  5   9
    2    b     a     e  3  4  11
    3    r     b     c  4  3   8
    4    u     b     d  5  2  10
    5    q     b     e  6  1  12
    """


class select_if(_if):
    """
    Select all columns that match a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    All sorted column names to uppercase

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select all sorted columns and convert names to upper case

    >>> def is_sorted(col):
    ...     a = col.values
    ...     return all(a[:-1] <= a[1:])
    >>> df >> select_if(is_sorted, str.upper)
      ALPHA  X
    0     a  1
    1     a  2
    2     a  3
    3     b  4
    4     b  5
    5     b  6

    Group columns are always selected.

    >>> df >> group_by('beta') >> select_if(is_sorted, str.upper)
    groups: ['beta']
      beta ALPHA  X
    0    b     a  1
    1    a     a  2
    2    b     a  3
    3    r     b  4
    4    u     b  5
    5    q     b  6
    """


class select_at(_at):
    """
    Select specific columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.

    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - String can be used for more complex
              statements, but the resulting names will be terrible.

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Select the listed columns and rename them to upper case.

    >>> df >> select_at(('alpha', 'x'), str.upper)
      ALPHA  X
    0     a  1
    1     a  2
    2     a  3
    3     b  4
    4     b  5
    5     b  6

    Select columns that contain the string ``eta`` and rename
    them name to upper case.

    >>> df >> select_at(dict(contains='eta'), str.upper)
      BETA THETA
    0    b     c
    1    a     d
    2    b     e
    3    r     c
    4    u     d
    5    q     e

    Group columns are always selected.

    >>> df >> group_by('beta') >> select_at(('alpha', 'x'), str.upper)
    groups: ['beta']
      beta ALPHA  X
    0    b     a  1
    1    a     a  2
    2    b     a  3
    3    r     b  4
    4    u     b  5
    5    q     b  6
    """


class summarize_all(_all):
    """
    Summarise all non-grouping columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - You can use this to access the aggregation
              functions provided in :class:`summarize`::

                  # Those that accept a single argument.
                  'min'
                  'max'
                  'sum'
                  'cumsum'
                  'mean'
                  'median'
                  'std'
                  'first'
                  'last'
                  'n_distinct'
                  'n_unique'

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    A single summarizing function

    >>> df >> select('x', 'z') >> summarize_all('mean')
         x    z
    0  3.5  9.5

    More than one summarizing function (as a tuple).

    >>> df >> select('x', 'z') >> summarize_all(('mean', np.std))
       x_mean  z_mean     x_std     z_std
    0     3.5     9.5  1.707825  1.707825

    You can use a dictionary to change postscripts of the
    column names.

    >>> (df
    ...  >> select('x', 'z')
    ...  >> summarize_all(dict(MEAN='mean', STD=np.std)))
       x_MEAN  z_MEAN     x_STD     z_STD
    0     3.5     9.5  1.707825  1.707825

    Group by

    >>> (df
    ...  >> group_by('alpha')
    ...  >> select('x', 'z')
    ...  >> summarize_all(('mean', np.std)))
      alpha  x_mean  z_mean     x_std     z_std
    0     a     2.0     9.0  0.816497  1.632993
    1     b     5.0    10.0  0.816497  1.632993

    Passing additional arguments

    >>> (df
    ...  >> group_by('alpha')
    ...  >> select('x', 'z')
    ...  >> summarize_all(np.std, ddof=1))
      alpha    x    z
    0     a  1.0  2.0
    1     b  1.0  2.0

    The arguments are passed to all functions, so in majority of
    these cases it might only be possible to summarise with one
    function.

    The group columns is never summarised.

    >>> (df
    ...  >> select('x', 'y', 'z')
    ...  >> define(parity='x%2')
    ...  >> group_by('parity')
    ...  >> summarize_all('mean'))
       parity    x    y         z
    0       1  3.0  4.0  9.333333
    1       0  4.0  3.0  9.666667
    """


class summarize_if(_if):
    """
    Summarise all columns that are true for a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    predicate : function or str
        A predicate function to be applied to the columns of the
        dataframe. Good candidates for predicate functions are
        those that check the type of the column. Such function
        are avaible at :mod:`pandas.api.dtypes`, for example
        :func:`pandas.api.types.is_numeric_dtype`.

        For convenience, you can reference the ``is_*_dtype``
        functions with shorter strings::

            'is_bool'             # pandas.api.types.is_bool_dtype
            'is_categorical'      # pandas.api.types.is_categorical_dtype
            'is_complex'          # pandas.api.types.is_complex_dtype
            'is_datetime64_any'   # pandas.api.types.is_datetime64_any_dtype
            'is_datetime64'       # pandas.api.types.is_datetime64_dtype
            'is_datetime64_ns'    # pandas.api.types.is_datetime64_ns_dtype
            'is_datetime64tz'     # pandas.api.types.is_datetime64tz_dtype
            'is_float'            # pandas.api.types.is_float_dtype
            'is_int64'            # pandas.api.types.is_int64_dtype
            'is_integer'          # pandas.api.types.is_integer_dtype
            'is_interval'         # pandas.api.types.is_interval_dtype
            'is_numeric'          # pandas.api.types.is_numeric_dtype
            'is_object'           # pandas.api.types.is_object_dtype
            'is_period'           # pandas.api.types.is_period_dtype
            'is_signed_integer'   # pandas.api.types.is_signed_integer_dtype
            'is_string'           # pandas.api.types.is_string_dtype
            'is_timedelta64'      # pandas.api.types.is_timedelta64_dtype
            'is_timedelta64_ns'   # pandas.api.types.is_timedelta64_ns_dtype
            'is_unsigned_integer' # pandas.api.types.is_unsigned_integer_dtype

        No other string values are allowed.

    functions : str or tuple or dict, optional
        Expressions or ``(name, expression)`` pairs. This should
        be used when the *name* is not a valid python variable
        name. The expression should be of type :class:`str` or
        an *interable* with the same number of elements as the
        dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandas.api.types as pdtypes
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Summarizing all numeric columns

    >>> df >> summarize_if(pdtypes.is_numeric_dtype, (np.min, np.max))
       x_amin  y_amin  z_amin  x_amax  y_amax  z_amax
    0       1       1       7       6       6      12

    Group by

    >>> (df
    ...  >> group_by('alpha')
    ...  >> summarize_if(pdtypes.is_numeric_dtype, (np.min, np.max))
    ... )
      alpha  x_amin  y_amin  z_amin  x_amax  y_amax  z_amax
    0     a       1       4       7       3       6      11
    1     b       4       1       8       6       3      12

    Using a ``'is_string'`` as a shortcut to :py:`pdtypes.is_string_dtype`
    for the predicate and custom summarizing a function.

    >>> def first(col): return list(col)[0]
    >>> df >> group_by('alpha') >> summarize_if('is_string', first)
      alpha beta theta
    0     a    b     c
    1     b    r     c

    Note, if the any of the group columns match the predictate, they
    are selected.
    """


class summarize_at(_at):
    """
    Summarize select columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names : tuple or dict
        Names of columns in dataframe. If a tuple, they should be names
        of columns. If a :class:`dict`, they keys must be in.

        - startswith : str or tuple, optional
            All column names that start with this string will be included.
        - endswith : str or tuple, optional
            All column names that end with this string will be included.
        - contains : str or tuple, optional
            All column names that contain with this string will be included.
        - matches : str or regex or tuple, optional
            All column names that match the string or a compiled regex pattern
            will be included. A tuple can be used to match multiple regexs.
        - drop : bool, optional
            If ``True``, the selection is inverted. The unspecified/unmatched
            columns are returned instead. Default is ``False``.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - The name (``__name__``) of the
              function is postfixed to resulting column names.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns, with the names determined as above.
            - :class:`dict` of the form ``{'name': function}`` - Allows
              you to apply one or more functions and also control the
              postfix to the name.
            - :class:`str` - You can use this to access the aggregation
              functions provided in :class:`summarize`::

                  # Those that accept a single argument.
                  'min'
                  'max'
                  'sum'
                  'cumsum'
                  'mean'
                  'median'
                  'std'
                  'first'
                  'last'
                  'n_distinct'
                  'n_unique'

    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    One variable

    >>> df >> summarize_at('x', ('mean', np.std))
       x_mean     x_std
    0     3.5  1.707825

    Many variables

    >>> df >> summarize_at(('x', 'y', 'z'), ('mean', np.std))
       x_mean  y_mean  z_mean     x_std     y_std     z_std
    0     3.5     3.5     9.5  1.707825  1.707825  1.707825

    Group by and many variables

    >>> (df
    ...  >> group_by('theta')
    ...  >> summarize_at(('x', 'y', 'z'), ('mean', np.std))
    ... )
      theta  x_mean  y_mean  z_mean  x_std  y_std  z_std
    0     c     2.5     4.5     7.5    1.5    1.5    0.5
    1     d     3.5     3.5     9.5    1.5    1.5    0.5
    2     e     4.5     2.5    11.5    1.5    1.5    0.5

    Using `select` parameters

    >>> (df
    ...  >> group_by('alpha')
    ...  >> summarize_at(
    ...         dict(endswith='ta'),
    ...         dict(unique_count=lambda col: len(pd.unique(col)))
    ...     )
    ... )
      alpha  beta_unique_count  theta_unique_count
    0     a                  2                   3
    1     b                  3                   3

    For this data, we can achieve the same using :class:`summarize`.

    >>> (df
    ...  >> group_by('alpha')
    ...  >> summarize(
    ...         beta_unique_count='len(pd.unique(beta))',
    ...         theta_unique_count='len(pd.unique(theta))'
    ...     )
    ... )
      alpha  beta_unique_count  theta_unique_count
    0     a                  2                   3
    1     b                  3                   3
    """


# Multiple Table Verbs


class _join(DoubleDataOperator):
    """
    Base class for join verbs
    """

    def __init__(self, x, y, on=None, left_on=None, right_on=None,
                 suffixes=('_x', '_y')):
        self.x = x
        self.y = y
        self.kwargs = dict(on=on, left_on=left_on, right_on=right_on,
                           suffixes=suffixes)


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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs
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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs
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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs
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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs
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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs

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

    def __init__(self, x, y, on=None, left_on=None, right_on=None):
        super().__init__(x, y, on=on, left_on=left_on, right_on=right_on)


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
        Columns on which to join. Must be found in both DataFrames.
    left_on : label or list, or array-like
        Field names to join on in left DataFrame. Can be a vector
        or list of vectors of the length of the DataFrame to use a
        particular vector as the join key instead of columns
    right_on : label or list, or array-like
        Field names to join on in right DataFrame or vector/list of
        vectors per left_on docs
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
    def __init__(self, x, y, on=None, left_on=None, right_on=None):
        super().__init__(x, y, on=on, left_on=left_on, right_on=right_on)


# Aliases
mutate = define
transmute = create
unique = distinct
summarise = summarize
full_join = outer_join
summarise_all = summarize_all
summarise_at = summarize_at
summarise_if = summarize_if
transmute_all = create_all
transmute_at = create_at
transmute_if = create_if
