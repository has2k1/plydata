"""
Tidy verb initializations
"""
from .operators import DataOperator
from .one_table_verbs import select

__all__ = ['gather']


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
