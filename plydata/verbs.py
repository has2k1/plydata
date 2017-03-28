"""
Verb Initializations
"""
import itertools

from .eval import EvalEnvironment
from .operators import DataOperator

__all__ = ['mutate', 'transmute', 'sample_n', 'sample_frac', 'select',
           'rename', 'distinct', 'unique', 'arrange', 'group_by',
           'summarize', 'summarise']


class mutate(DataOperator):
    """
    Add column to DataFrame

    Parameters
    ----------
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
    >>> df >> mutate(x_sq='x**2')
       x  x_sq
    0  1     1
    1  2     4
    2  3     9
    >>> df >> mutate(('x*2', 'x*2'), ('x*3', 'x*3'), x_cubed='x**3')
       x  x_sq  x*2  x*3  x_cubed
    0  1     1    2    3        1
    1  2     4    4    6        8
    2  3     9    6    9       27
    >>> df >> mutate('x*4')
       x  x_sq  x*2  x*3  x_cubed  x*4
    0  1     1    2    3        1    4
    1  2     4    4    6        8    8
    2  3     9    6    9       27   12
    """
    new_columns = None
    expressions = None  # Expressions to create the new columns

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cols = []
        exprs = []
        for arg in args:
            if isinstance(arg, str):
                col = expr = arg
            else:
                col, expr = arg
            cols.append(col)
            exprs.append(col)
        self.new_columns = itertools.chain(cols, kwargs.keys())
        self.expressions = itertools.chain(exprs, kwargs.values())


class transmute(mutate):
    """
    Create DataFrame with columns

    Similar to :class:`mutate`, but it drops the existing columns.

    Parameters
    ----------
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
    >>> df >> transmute(x_sq='x**2')
       x_sq
    0     1
    1     4
    2     9
    >>> df >> transmute(('x*2', 'x*2'), ('x*3', 'x*3'), x_cubed='x**3')
       x*2  x*3  x_cubed
    0    2    3        1
    1    4    6        8
    2    6    9       27
    >>> df >> transmute('x*4')
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
    args : tuple, optional
        A single positional argument that holds
        ``{'old_name': 'new_name'}`` pairs. This is useful if the
        *old_name* is not a valid python variable name.
    kwargs : dict, optional
        ``{old_name: 'new_name'}`` pairs. If all the columns to be
        renamed are valid python variable names, then they
        can be specified as keyword arguments.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> df = pd.DataFrame({'bell': x, 'whistle': x,
    ...                    'nail': x, 'tail': x})
    >>> df >> rename(bell='gong', nail='pin')
       gong  pin  tail  whistle
    0     1    1     1        1
    1     2    2     2        2
    2     3    3     3        3
    >>> df >> rename({'tail': 'flap'}, nail='pin')
       bell  pin  flap  whistle
    0     1    1     1        1
    1     2    2     2        2
    2     3    3     3        3
    """
    lookup = None

    def __init__(self, *args, **kwargs):
        lookup = args[0] if len(args) else {}
        self.lookup = lookup.copy()
        self.lookup.update(kwargs)


class distinct(DataOperator):
    """
    Select distinct/unique rows

    Parameters
    ----------
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
    >>> df >> mutate(z='x%2') >> distinct(['x', 'z'])
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
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            if isinstance(args[0], str):
                self.keep = args[0]
            else:
                self.columns = args[0]
        elif len(args) == 2:
            self.columns, self.keep = args
        elif len(args) > 2:
            raise Exception("Too many positional arguments.")

        # Mutate
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
        super().__init__(*args)
        self.expressions = [x for x in args]


class group_by(mutate):
    """
    Group dataframe by one or more columns/variables

    Parameters
    ----------
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
    <BLANKLINE>
       x  y
    0  1  1
    1  5  2
    2  2  3
    3  2  4
    4  4  5
    5  0  6
    6  4  5

    Like :meth:`mutate`, :meth:`group_by` creates any
    missing columns.

    >>> df >> group_by('y-1', xplus1='x+1')
    groups: ['y-1', 'xplus1']
    <BLANKLINE>
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
    <BLANKLINE>
       y  y-1  xplus1
    0  1    0       2
    1  2    1       6
    2  3    2       3
    3  4    3       3
    4  5    4       5
    5  6    5       1
    6  5    4       5
    """
    groups = None

    def __init__(self, *args, **kwargs):
        self.env = EvalEnvironment.capture(1)
        super().__init__(*args, **kwargs)
        self.new_columns = list(self.new_columns)
        self.groups = list(self.new_columns)


class summarize(DataOperator):
    """
    Summarise multiple values to a single value.

    Parameters
    ----------
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
    """
    new_columns = None
    epressions = None  # Expressions to create the summary columns

    def __init__(self, *args, **kwargs):
        self.env = EvalEnvironment.capture(1)
        cols = []
        exprs = []
        for arg in args:
            if isinstance(arg, str):
                col = expr = arg
            else:
                col, expr = arg
            cols.append(col)
            exprs.append(col)

        self.new_columns = list(itertools.chain(cols, kwargs.keys()))
        self.expressions = list(itertools.chain(exprs, kwargs.values()))


# Multiple Table Verbs


# Aliases
unique = distinct
summarise = summarize
