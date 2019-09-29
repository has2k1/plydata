"""
Two table verb initializations
"""

from .operators import DoubleDataOperator

__all__ = ['inner_join', 'outer_join', 'left_join', 'right_join',
           'full_join', 'anti_join', 'semi_join']


class _join(DoubleDataOperator):
    """
    Base class for join verbs
    """

    def __init__(self, *args, on=None, left_on=None, right_on=None,
                 suffixes=('_x', '_y')):
        if len(args) == 2:
            self.x, self.y = args
        elif len(args) == 1:
            self.x, self.y = None, args[0]
        else:
            tpl = "{} cannot take more than two positional arguments"
            raise ValueError(tpl.format(self.__class__.__name__))

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

    Notes
    -----
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

    Notes
    -----
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

    Notes
    -----
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

    Notes
    -----
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

    Notes
    -----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """

    def __init__(self, *args, on=None, left_on=None, right_on=None):
        super().__init__(*args, on=on, left_on=left_on, right_on=right_on)


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

    Notes
    -----
    Groups are ignored for the purpose of joining, but the result
    preserves the grouping of x.
    """
    def __init__(self, *args, on=None, left_on=None, right_on=None):
        super().__init__(*args, on=on, left_on=left_on, right_on=right_on)


full_join = outer_join
