"""
Tidy verb initializations
"""
import re

import pandas.api.types as pdtypes

from plydata.operators import DataOperator
from plydata.one_table_verbs import select
from plydata.utils import verify_arg, hasattrs, mean_if_many

__all__ = [
    'gather',
    'spread',
    'separate',
    'extract',
    'pivot_wider',
]


class gather(DataOperator):
    """
    Collapse multiple columns into key-value pairs.

    You use gather() when you notice that you have columns
    that are not variables.

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    key : str
        Name of the variable column
    value : str
        Name of the value column
    *columns : list-like | select | str | slice
        Columns to be gathered and whose contents will
        make values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['mary', 'oscar', 'martha', 'john'],
    ...     'math': [92, 83, 85, 90],
    ...     'art': [75, 95, 80, 72]
    ... })
    >>> df
         name  math  art
    0    mary    92   75
    1   oscar    83   95
    2  martha    85   80
    3    john    90   72
    >>> df >> gather('subject', 'grade', ['math', 'art'])
         name subject  grade
    0    mary    math     92
    1   oscar    math     83
    2  martha    math     85
    3    john    math     90
    4    mary     art     75
    5   oscar     art     95
    6  martha     art     80
    7    john     art     72

    You can use the :class:`~plydata.select` verb to choose the columns

    >>> df >> gather('subject', 'grade', select('math', 'art'))
         name subject  grade
    0    mary    math     92
    1   oscar    math     83
    2  martha    math     85
    3    john    math     90
    4    mary     art     75
    5   oscar     art     95
    6  martha     art     80
    7    john     art     72
    >>> df >> gather('subject', 'grade', select('-name'))
         name subject  grade
    0    mary    math     92
    1   oscar    math     83
    2  martha    math     85
    3    john    math     90
    4    mary     art     75
    5   oscar     art     95
    6  martha     art     80
    7    john     art     72

    ``gather('subject', 'grade', '-name')`` without the ``select``
    would have worked.

    >>> df >> gather('subject', 'grade', select(startswith='m'))
         name  art subject  grade
    0    mary   75    math     92
    1   oscar   95    math     83
    2  martha   80    math     85
    3    john   72    math     90
    """

    # Powers the column selection
    _select_verb = None

    def __init__(self, key, value, *columns):
        self.key = key
        self.value = value
        self._select_verb = select.from_columns(*columns)


class spread(DataOperator):
    """
    Spread a key-value pair across multiple columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    key : str
        Name of the variable column
    value : str
        Name of the value column
    sep : str
        Charater(s) used to separate the column names. This is used
        to add a hierarchy and resolve duplicate column names.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['mary', 'oscar', 'martha', 'john'] * 2,
    ...     'subject': np.repeat(['math', 'art'], 4),
    ...     'grade': [92, 83, 85, 90, 75, 95, 80, 72]
    ... })
    >>> df
         name subject  grade
    0    mary    math     92
    1   oscar    math     83
    2  martha    math     85
    3    john    math     90
    4    mary     art     75
    5   oscar     art     95
    6  martha     art     80
    7    john     art     72
    >>> df >> spread('subject', 'grade')
         name  art  math
    0    john   72    90
    1  martha   80    85
    2    mary   75    92
    3   oscar   95    83
    """

    def __init__(self, key, value, sep='_'):
        self.key = key
        self.value = value
        self.sep = sep


class separate(DataOperator):
    r"""
    Split a single column into multiple columns

    Parameters
    ----------
    col : str | int
        Column name or position of variable to separate.
    into : list-like
        Column names. Use ``None`` to omit the variable from the
        output.
    sep : str | regex | list-like
        If String or regex, it is the pattern at which to separate
        the strings in the column. The default value separates on
        a string of non-alphanumeric characters.

        If list-like it must contain positions to split at. The length
        of the list should be 1 less than ``into``.
    remove : bool
        If ``True`` remove input column from output frame.
    convert : bool
        If ``True`` convert result columns to int, float or bool
        where appropriate.
    extra : 'warn' | 'drop' | 'merge'
        Control what happens when there are too many pieces. Only
        applies if ``sep`` is a string/regex.

        - 'warn'(the default): warn and drop extra values.
        - 'drop': drop any extra values without a warning.
        - 'merge': only splits at most :py:`len(into)` times.

    fill : 'warn' | 'right' | 'left'
        Control what happens when there are not enough pieces. Only
        applies if ``sep`` is a string/regex.

        - 'warn' (the default): warn and fill from the right
        - 'right': fill with missing values on the right
        - 'left': fill with missing values on the left

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a,1', 'b,2', 'c,3'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha    x  zeta
    0      1  a,1     6
    1      1  b,2     6
    2      1  c,3     6
    >>> df >> separate('x', into=['A', 'B'])
       alpha  A  B  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6
    >>> df >> separate('x', into=['A', 'B'], remove=False)
       alpha    x  A  B  zeta
    0      1  a,1  a  1     6
    1      1  b,2  b  2     6
    2      1  c,3  c  3     6

    Using an array of positions and using ``None`` to omit a
    variable.

    >>> df >> separate('x', into=['A', None, 'C'], sep=(1, 2))
       alpha  A  C  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6

    Dealing with extra pieces

    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a,1', 'b,2', 'c,3,d'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha      x  zeta
    0      1    a,1     6
    1      1    b,2     6
    2      1  c,3,d     6
    >>> df >> separate('x', into=['A', 'B'], extra='merge')
       alpha  A    B  zeta
    0      1  a    1     6
    1      1  b    2     6
    2      1  c  3,d     6
    >>> df >> separate('x', into=['A', 'B'], extra='drop')
       alpha  A  B  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6

    Dealing with fewer pieces

    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a,1', 'b,2', 'c'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha    x  zeta
    0      1  a,1     6
    1      1  b,2     6
    2      1    c     6
    >>> df >> separate('x', into=['A', 'B'], fill='right')
       alpha  A     B  zeta
    0      1  a     1     6
    1      1  b     2     6
    2      1  c  None     6
    >>> df >> separate('x', into=['A', 'B'], fill='left')
       alpha     A  B  zeta
    0      1     a  1     6
    1      1     b  2     6
    2      1  None  c     6

    Missing values

    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a,1', None, 'c,3'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha     x  zeta
    0      1   a,1     6
    1      1  None     6
    2      1   c,3     6
    >>> df >> separate('x', into=['A', 'B'])
       alpha     A     B  zeta
    0      1     a     1     6
    1      1  None  None     6
    2      1     c     3     6

    More than one character separators. Any spaces must be
    included in the separator

    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a -> 1', 'b -> 2', 'c -> 3'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha       x  zeta
    0      1  a -> 1     6
    1      1  b -> 2     6
    2      1  c -> 3     6
    >>> df >> separate('x', into=['A', 'B'], sep=' -> ')
       alpha  A  B  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6

    All values of ``sep`` are treated ad regular expression, but a
    compiled regex can is also permitted.

    >>> pattern = re.compile(r'\s*->\s*')
    >>> df >> separate('x', into=['A', 'B'], sep=pattern)
       alpha  A  B  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6

    """
    # Use _pattern when sep is a string/regex
    # Use _positions when sep is a list of split positions
    _pattern = None
    _positions = None

    def __init__(
            self,
            col,
            into,
            sep=r'[^A-Za-z0-9]+',
            remove=True,
            convert=False,
            extra='warn',
            fill='warn'
    ):
        verify_arg(extra, 'extra', ('warn', 'drop', 'merge'))
        verify_arg(fill, 'fill', ('warn', 'right', 'left'))

        if isinstance(sep, str):
            self._pattern = re.compile(sep)
        elif hasattrs(sep, ('split', 'match')):
            self._pattern = sep
        else:
            n, n_ref = len(sep), len(into) - 1
            if n != n_ref:
                raise ValueError(
                    "length of `sep` is {}, it should be {}, 1 less "
                    "than `into` (the number of pieces).".format(
                        n, n_ref
                    ))
            self._positions = [0] + list(sep) + [None]

        self.col = col
        self.into = into
        self.sep = sep
        self.remove = remove
        self.convert = convert
        self.extra = extra
        self.fill = fill


class extract(DataOperator):
    r"""
    Split a column using a regular expression with capturing groups.

    If the groups don't match, or the input is NA, the output will be NA.

    Parameters
    ----------
    col : str | int
        Column name or position of variable to separate.
    into : list-like
        Column names. Use ``None`` to omit the variable from the
        output.
    regex : str | regex
        Pattern used to extract columns from ``col``. There should be
        only one group (defined by ``()``) for each element of ``into``.
    remove : bool
        If ``True`` remove input column from output frame.
    convert : bool
        If ``True`` convert result columns to int, float or bool
        where appropriate.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'alpha': 1,
    ...    'x': ['a,1', 'b,2', 'c,3'],
    ...    'zeta': 6
    ... })
    >>> df
       alpha    x  zeta
    0      1  a,1     6
    1      1  b,2     6
    2      1  c,3     6
    >>> df >> extract('x', into='A')
       alpha  A  zeta
    0      1  a     6
    1      1  b     6
    2      1  c     6
    >>> df >> extract('x', into=['A', 'B'], regex=r'(\w+),(\w+)')
       alpha  A  B  zeta
    0      1  a  1     6
    1      1  b  2     6
    2      1  c  3     6

    >>> df >> extract('x', into=['A', 'B'], regex=r'(\w+),(\w+)', remove=False)
       alpha    x  A  B  zeta
    0      1  a,1  a  1     6
    1      1  b,2  b  2     6
    2      1  c,3  c  3     6

    Convert extracted columns to appropriate data types.

    >>> result = df >> extract(
    ...    'x', into=['A', 'B'], regex=r'(\w+),(\w+)', convert=True)
    >>> result['B'].dtype
    dtype('int64')

    The regex must match fully, not just the individual groups.

    >>> df >> extract('x', into=['A', 'B'], regex=r'(\w+),([12]+)')
       alpha    A    B  zeta
    0      1    a    1     6
    1      1    b    2     6
    2      1  NaN  NaN     6
    """

    def __init__(
            self,
            col,
            into,
            regex=r'([A-Za-z0-9]+)',
            remove=True,
            convert=False,
    ):
        if isinstance(regex, str):
            self.regex = re.compile(regex)
        elif hasattrs(regex, ('split', 'match')):
            self.regex = regex
        else:
            raise TypeError(
                "Unknown type `{}` used to describe a regular "
                "expression.".format(type(regex))
            )
        if not pdtypes.is_list_like(into):
            into = [into]

        self.col = col
        self.into = into
        self.regex = regex
        self.remove = remove
        self.convert = convert


class pivot_wider(DataOperator):
    """
    Spread a key-value pair across multiple columns

    Parameters
    ----------
    data : dataframe, optional
        Useful when not using the ``>>`` operator.
    names_from : str
        Column where to get the wide column names.
    values_from : str | list-like
        Column(s) where to get observation values that will
        be placed in the wide columns.
    id_cols : list-like
        Columns that uniquely identify each observation. The default
        is all columns in the data except those in ``names_from`` and
        ``values_from``. Typically used when you have additional
        variables that is directly related.
    names_prefix : str
        String added to the start of every variable name. This is
        particularly useful if ``names_from`` is a numeric vector and you
        want to create syntactic variable names.
    names_sep : str
        If ``names_from`` or ``values_from`` contains multiple variables,
        with the same name this will be used to join their values together
        into a single string to use as a column name.
    values_fill : object
        What to fill in if there are missing values.j
    values_fn : callable | dict
        A function to be applied to each cell if  ``names_from`` &
        ``values_from`` or ``id_cols`` do not uniquely identify an
        observation. The function is used to aggregate the multiple
        observations. A ``dict`` can be used to apply a different
        function to each of the columns. The default is to compute
        the mean for all.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['mary', 'oscar', 'martha', 'john'] * 2,
    ...     'initials': ['M.K', 'O.S', 'M.J', 'J.T'] * 2,
    ...     'subject': np.repeat(['math', 'art'], 4),
    ...     'grade': [92, 83, 85, 90, 75, 95, 80, 72],
    ...     'midterm': [88, 83, 89, 93, 85, 95, 76, 79]
    ... })
    >>> df
         name initials subject  grade  midterm
    0    mary      M.K    math     92       88
    1   oscar      O.S    math     83       83
    2  martha      M.J    math     85       89
    3    john      J.T    math     90       93
    4    mary      M.K     art     75       85
    5   oscar      O.S     art     95       95
    6  martha      M.J     art     80       76
    7    john      J.T     art     72       79
    >>> df >> pivot_wider(
    ...     names_from='subject',
    ...     values_from=('grade', 'midterm')
    ... )
      initials    name  grade_art  grade_math  midterm_art  midterm_math
    0      J.T    john         72          90           79            93
    1      M.J  martha         80          85           76            89
    2      M.K    mary         75          92           85            88
    3      O.S   oscar         95          83           95            83

    Be specific about the identifier column.

    >>> df >> pivot_wider(
    ...     names_from='subject',
    ...     values_from=('grade', 'midterm'),
    ...     id_cols='name'
    ... )
         name  grade_art  grade_math  midterm_art  midterm_math
    0    john         72          90           79            93
    1  martha         80          85           76            89
    2    mary         75          92           85            88
    3   oscar         95          83           95            83

    When there is no ambiguity about the column names, the previous
    column names are not prepended onto the new names

    >>> df >> pivot_wider(
    ...     names_from='subject',
    ...     values_from='grade',
    ...     id_cols='name'
    ... )
         name  art  math
    0    john   72    90
    1  martha   80    85
    2    mary   75    92
    3   oscar   95    83

    *Dealing with non-syntactic column names in the result*

    >>> np.random.seed(123)
    >>> df = pd.DataFrame({
    ...     'name': ['mary', 'oscar'] * 6,
    ...     'face': np.repeat([1, 2, 3, 4, 5, 6], 2),
    ...     'rolls': np.random.randint(5, 21, 12)
    ... })
    >>> df
         name  face  rolls
    0    mary     1     19
    1   oscar     1     18
    2    mary     2     19
    3   oscar     2      7
    4    mary     3     17
    5   oscar     3      7
    6    mary     4     11
    7   oscar     4      6
    8    mary     5      8
    9   oscar     5     15
    10   mary     6     16
    11  oscar     6     14
    >>> df >> pivot_wider(
    ...     names_from='face',
    ...     values_from='rolls'
    ... )
        name   1   2   3   4   5   6
    0   mary  19  19  17  11   8  16
    1  oscar  18   7   7   6  15  14
    >>> df >> pivot_wider(
    ...     names_from='face',
    ...     values_from='rolls',
    ...     names_prefix='rolled_',
    ... )
        name  rolled_1  rolled_2  rolled_3  rolled_4  rolled_5  rolled_6
    0   mary        19        19        17        11         8        16
    1  oscar        18         7         7         6        15        14
    """

    def __init__(
            self,
            names_from,
            values_from,
            id_cols=None,
            names_prefix='',
            names_sep='_',
            values_fill=None,
            values_fn=mean_if_many
    ):
        def as_list_like(x):
            if x is not None and not pdtypes.is_list_like(x):
                return [x]
            return x

        self.id_cols = as_list_like(id_cols)
        self.names_from = as_list_like(names_from)
        self.values_from = as_list_like(values_from)
        self.names_prefix = names_prefix
        self.names_sep = names_sep
        self.values_fill = values_fill
        self.values_fn = values_fn
