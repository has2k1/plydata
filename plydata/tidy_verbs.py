"""
Tidy verb initializations
"""
import re

from .operators import DataOperator
from .one_table_verbs import select
from .utils import verify_arg, hasattrs

__all__ = ['gather', 'spread', 'separate']


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
