"""
Helper verbs
"""
from .operators import DataOperator
from .one_table_verbs import select, group_by


__all__ = ['call', 'tally', 'count', 'add_tally', 'add_count',
           'arrange_all', 'arrange_at', 'arrange_if',
           'create_all', 'create_at', 'create_if',
           'group_by_all', 'group_by_at', 'group_by_if',
           'mutate_all', 'mutate_at', 'mutate_if',
           'query_all', 'query_at', 'query_if',
           'rename_all', 'rename_at', 'rename_if',
           'select_all', 'select_at', 'select_if',
           'summarize_all', 'summarize_at', 'summarize_if',
           # Aliases
           'summarise_all', 'summarise_at', 'summarise_if',
           'transmute_all', 'transmute_at', 'transmute_if',
           ]

MANY = float('inf')


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
    >>> from plydata import *
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
    >>> from plydata import tally, group_by, summarize
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

    You can do that with :class:`~plydata.verbs.summarize`

    >>> df >> group_by('y') >> summarize(n='sum(x*w)')
       y  n
    0  a  9
    1  b 24
    """

    def __init__(self, weights=None, sort=False):
        self.set_env_from_verb_init()
        self.weights = weights
        self.sort = sort


class count(group_by):
    """
    Count observations by group

    ``count`` is a convenient wrapper for summarise that will
    either call n or sum(n) depending on whether you’re
    tallying for the first time, or re-tallying. Similar to
    :class:`tally`, but it does the :class:`~plydata.verbs.group_by`
    for you.

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
    >>> from plydata import count, group_by, summarize
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

    You can do that with :class:`~plydata.verbs.summarize`

    >>> df >> group_by('y') >> summarize(n='sum(x*w)')
       y  n
    0  a  9
    1  b 24
    """

    def __init__(self, *args, weights=None, sort=False):
        self.set_env_from_verb_init()
        super().__init__(*args)
        self.add_ = True
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
    >>> from plydata import *
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': ['a', 'b', 'a', 'b', 'a', 'b'],
    ...     'w': [1, 2, 1, 2, 1, 2]})

    Without groups it is one large group

    >>> df >> add_tally()
       x  y  w  n
    0  1  a  1  6
    1  2  b  2  6
    2  3  a  1  6
    3  4  b  2  6
    4  5  a  1  6
    5  6  b  2  6

    Sum of the weights

    >>> df >> add_tally('w')
       x  y  w  n
    0  1  a  1  9
    1  2  b  2  9
    2  3  a  1  9
    3  4  b  2  9
    4  5  a  1  9
    5  6  b  2  9

    With groups

    >>> df >> group_by('y') >> add_tally()
    groups: ['y']
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  3
    2  3  a  1  3
    3  4  b  2  3
    4  5  a  1  3
    5  6  b  2  3

    With groups and weights

    >>> df >> group_by('y') >> add_tally('w')
    groups: ['y']
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  6
    2  3  a  1  3
    3  4  b  2  6
    4  5  a  1  3
    5  6  b  2  6

    Applying the weights to a column

    >>> df >> group_by('y') >> add_tally('x*w')
    groups: ['y']
       x  y  w   n
    0  1  a  1   9
    1  2  b  2  24
    2  3  a  1   9
    3  4  b  2  24
    4  5  a  1   9
    5  6  b  2  24

    Add tally is equivalent to using :func:`sum` or ``n()``
    in :class:`~plydata.verbs.define`.

    >>> df >> group_by('y') >> define(n='sum(x*w)')
    groups: ['y']
       x  y  w   n
    0  1  a  1   9
    1  2  b  2  24
    2  3  a  1   9
    3  4  b  2  24
    4  5  a  1   9
    5  6  b  2  24

    >>> df >> group_by('y') >> define(n='n()')
    groups: ['y']
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  3
    2  3  a  1  3
    3  4  b  2  3
    4  5  a  1  3
    5  6  b  2  3

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
    >>> from plydata import *
    >>> df = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': ['a', 'b', 'a', 'b', 'a', 'b'],
    ...     'w': [1, 2, 1, 2, 1, 2]})

    Without groups it is one large group

    >>> df >> add_count()
       x  y  w  n
    0  1  a  1  6
    1  2  b  2  6
    2  3  a  1  6
    3  4  b  2  6
    4  5  a  1  6
    5  6  b  2  6

    Sum of the weights

    >>> df >> add_count(weights='w')
       x  y  w  n
    0  1  a  1  9
    1  2  b  2  9
    2  3  a  1  9
    3  4  b  2  9
    4  5  a  1  9
    5  6  b  2  9

    With groups

    >>> df >> add_count('y')
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  3
    2  3  a  1  3
    3  4  b  2  3
    4  5  a  1  3
    5  6  b  2  3

    >>> df >> group_by('y') >> add_count()
    groups: ['y']
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  3
    2  3  a  1  3
    3  4  b  2  3
    4  5  a  1  3
    5  6  b  2  3

    With groups and weights

    >>> df >> add_count('y', weights='w')
       x  y  w  n
    0  1  a  1  3
    1  2  b  2  6
    2  3  a  1  3
    3  4  b  2  6
    4  5  a  1  3
    5  6  b  2  6

    Applying the weights to a column

    >>> df >> add_count('y', weights='x*w')
       x  y  w   n
    0  1  a  1   9
    1  2  b  2  24
    2  3  a  1   9
    3  4  b  2  24
    4  5  a  1   9
    5  6  b  2  24

    You can do that with :class:`add_tally`

    >>> df >> group_by('y') >> add_tally('x*w') >> ungroup()
       x  y  w   n
    0  1  a  1   9
    1  2  b  2  24
    2  3  a  1   9
    3  4  b  2  24
    4  5  a  1   9
    5  6  b  2  24

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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    n_functions = MANY   # Maximum number of functions

    def __init__(self, functions=None, *args, **kwargs):
        if functions is None:
            functions = (lambda x: x, )
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        n = len(functions)

        if n > self.n_functions:
            raise ValueError(
                "{} expected {} function(s) got {}".format(
                    self.__class__.__name__, self.n_functions, n
                )
            )

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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    n_functions = MANY   # Maximum number of functions

    def __init__(self, predicate, functions=None, *args, **kwargs):
        if functions is None:
            functions = (lambda x: x, )
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        n = len(functions)

        if n > self.n_functions:
            raise ValueError(
                "{} expected {} function(s) got {}".format(
                    self.__class__.__name__, self.n_functions, n
                )
            )

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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    n_functions = MANY   # Maximum number of functions

    def __init__(self, names, functions=None, *args, **kwargs):
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
            functions = (lambda x: x, )
        elif isinstance(functions, str) or callable(functions):
            functions = (functions,)
        elif isinstance(functions, dict):
            functions = functions
        else:
            functions = tuple(functions)

        n = len(functions)

        if n > self.n_functions:
            raise ValueError(
                "{} expected {} function(s) got {}".format(
                    self.__class__.__name__, self.n_functions, n
                )
            )

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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

    Notes
    -----
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

    Notes
    -----
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    Arrange by explictily naming the columns to arrange by.
    This is not much different from :class:`~plydata.verbs.arrange`.

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

    Notes
    -----
    Do not use functions that change the order of the values in the
    array. Such functions are most likely the wrong candidates,
    they corrupt the data. Use function(s) that return values that
    can be sorted.
    """
    def __init__(self, names, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(names, functions, *args, **kwargs)


class create_all(_all):
    """
    Create a new dataframe with all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
    >>> df = pd.DataFrame({
    ...     'alpha': list('aaabbb'),
    ...     'beta': list('babruq'),
    ...     'theta': list('cdecde'),
    ...     'x': [1, 2, 3, 4, 5, 6],
    ...     'y': [6, 5, 4, 3, 2, 1],
    ...     'z': [7, 9, 11, 8, 10, 12]
    ... })

    In the simplest form it is not too different from
    :class:`~plydata.verbs.group_by`.

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
    >>> df >> group_by_at(dict(matches=r'\\w+eta$'), double)
    groups: ['beta', 'theta']
      alpha beta theta  x  y   z
    0     a   bb    cc  1  6   7
    1     a   aa    dd  2  5   9
    2     a   bb    ee  3  4  11
    3     b   rr    cc  4  3   8
    4     b   uu    dd  5  2  10
    5     b   qq    ee  6  1  12
    """
    def __init__(self, names, functions=None, *args, **kwargs):
        self.set_env_from_verb_init()
        super().__init__(names, functions, *args, **kwargs)


class mutate_all(_all):
    """
    Modify all columns that are true for a predicate

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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
    >>> from plydata import *
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
    >>> from plydata import *
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
    >>> from plydata import *
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
    functions : callable
        Useful when not using the ``>>`` operator.

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
    >>> from plydata import *
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
    n_functions = 1


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
    functions : callable
        Useful when not using the ``>>`` operator.
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
    >>> from plydata import *
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
    n_functions = 1


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
    function : callable
        Function to rename the column(s).
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
    >>> from plydata import *
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
    n_functions = 1


class select_all(_all):
    """
    Select all columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    function : callable
        Function to rename the column(s).
    args : tuple
        Arguments to the functions. The arguments are pass to *all*
        functions.
    kwargs : dict
        Keyword arguments to the functions. The keyword arguments are
        passed to *all* functions.

    Examples
    --------
    >>> import pandas as pd
    >>> from plydata import *
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
    n_functions = 1


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
    function : callable
        Function to rename the column(s).
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
    >>> from plydata import *
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
    n_functions = 1


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

    function : callable
        Functions to rename the column(s).
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
    >>> from plydata import *
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
    n_functions = 1


class summarize_all(_all):
    """
    Summarise all non-grouping columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    functions : callable or tuple or dict or str
        Functions to alter the columns:

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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
    >>> from plydata import *
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

            - function (any callable) - Function is applied to the
              column and the result columns replace the original
              columns.
            - :class:`tuple` of functions - Each function is applied to
              all of the columns and the name (``__name__``) of the
              function is postfixed to resulting column names.
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
    >>> from plydata import *
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

    For this data, we can achieve the same using
    :class:`~plydata.verbs.summarize`.

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


# Aliases
summarise_all = summarize_all
summarise_at = summarize_at
summarise_if = summarize_if
transmute_all = create_all
transmute_at = create_at
transmute_if = create_if
