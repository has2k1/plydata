"""
Common functions and classes for implementing dataframe verbs
"""
import itertools
import re
from contextlib import suppress

import pandas as pd
import pandas.api.types as pdtypes
import numpy as np

from ..expressions import Expression
from ..types import GroupedDataFrame
from ..utils import get_empty_env


def _get_groups(verb):
    """
    Return groups
    """
    try:
        return verb.data.plydata_groups
    except AttributeError:
        return []


def _get_base_dataframe(df):
    """
    Remove all columns other than those grouped on
    """
    if isinstance(df, GroupedDataFrame):
        base_df = GroupedDataFrame(
            df.loc[:, df.plydata_groups], df.plydata_groups,
            copy=True)
    else:
        base_df = pd.DataFrame(index=df.index)
    return base_df


def _add_group_columns(data, gdf):
    """
    Add group columns to data with a value from the grouped dataframe

    It is assumed that the grouped dataframe contains a single group

    >>> data = pd.DataFrame({
    ...     'x': [5, 6, 7]})
    >>> gdf = GroupedDataFrame({
    ...     'g': list('aaa'),
    ...     'x': range(3)}, groups=['g'])
    >>> _add_group_columns(data, gdf)
       g  x
    0  a  5
    1  a  6
    2  a  7
    """
    n = len(data)
    if isinstance(gdf, GroupedDataFrame):
        for i, col in enumerate(gdf.plydata_groups):
            if col not in data:
                group_values = [gdf[col].iloc[0]] * n
                # Need to be careful and maintain the dtypes
                # of the group columns
                if pdtypes.is_categorical_dtype(gdf[col]):
                    col_values = pd.Categorical(
                        group_values,
                        categories=gdf[col].cat.categories,
                        ordered=gdf[col].cat.ordered
                    )
                else:
                    col_values = pd.Series(
                        group_values,
                        index=data.index,
                        dtype=gdf[col].dtype
                    )
                # Group columns come first
                data.insert(i, col, col_values)
    return data


def _create_column(data, col, value):
    """
    Create column in dataframe

    Helper method meant to deal with problematic
    column values. e.g When the series index does
    not match that of the data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe in which to insert value
    col : column label
        Column name
    value : object
        Value to assign to column

    Returns
    -------
    data : pandas.DataFrame
        Modified original dataframe

    >>> df = pd.DataFrame({'x': [1, 2, 3]})
    >>> y = pd.Series([11, 12, 13], index=[21, 22, 23])

    Data index and value index do not match

    >>> _create_column(df, 'y', y)
       x   y
    0  1  11
    1  2  12
    2  3  13

    Non-empty dataframe, scalar value

    >>> _create_column(df, 'z', 3)
       x   y  z
    0  1  11  3
    1  2  12  3
    2  3  13  3

    Empty dataframe, scalar value

    >>> df = pd.DataFrame()
    >>> _create_column(df, 'w', 3)
       w
    0  3
    >>> _create_column(df, 'z', 'abc')
       w    z
    0  3  abc
    """
    with suppress(AttributeError):
        # If the index of a series and the dataframe
        # in which the series will be assigned to a
        # column do not match, missing values/NaNs
        # are created. We do not want that.
        if not value.index.equals(data.index):
            if len(value) == len(data):
                value.index = data.index
            else:
                value.reset_index(drop=True, inplace=True)

    # You cannot assign a scalar value to a dataframe
    # without an index. You need an interable value.
    if data.index.empty:
        try:
            len(value)
        except TypeError:
            scalar = True
        else:
            scalar = isinstance(value, str)

        if scalar:
            value = [value]

    data[col] = value
    return data


class Evaluator:
    """
    Evaluator for expressions of a verb

    The Evaluator is responsible for breaking up the data,
    doing evaluations and putting the pieces together

    Parameters
    ----------
    verb : verb
        Verb with expressions to evaluate
    keep_index : bool
        If True, the evaluation method *tries* to create a
        dataframe with the same index as the original.
        Ultimately whether this succeeds depends on the
        expressions to be evaluated.
    keep_groups : bool
        If True, the resulting data will be grouped in the input
        data was grouped.
    drop : bool
        If True drop unselected columns. (Used by create)
    """
    def __init__(self, verb, keep_index=True, keep_groups=True, drop=False):
        self.data = verb.data
        self.expressions = verb.expressions
        self.env = get_empty_env() if verb.env is None else verb.env
        self.keep_index = keep_index
        self.keep_groups = keep_groups
        self.drop = drop

        try:
            self.groups = verb.data.plydata_groups
        except AttributeError:
            self.groups = None

    def process(self):
        """
        Run the expressions

        Returns
        -------
        out : pandas.DataFrame
            Resulting data
        """
        # Short cut
        if self._all_expressions_evaluated():
            if self.drop:
                # Drop extra columns. They do not correspond to
                # any expressions.
                columns = [expr.column for expr in self.expressions]
                self.data = self.data.loc[:, columns]
            return self.data

        # group_by
        # evaluate expressions
        # combine columns
        # concat evalutated group data and clean up index and group
        gdfs = self._get_group_dataframes()
        egdfs = self._evaluate_expressions(gdfs)
        edata = self._concat(egdfs)
        return edata

    def _all_expressions_evaluated(self):
        """
        Return True if all expressions match with the columns

        Saves some processor cycles
        """
        def present(expr):
            return (isinstance(expr.stmt, str) and
                    expr.stmt == expr.column and
                    expr.column in self.data)
        return all(present(expr) for expr in self.expressions)

    def _get_group_dataframes(self):
        """
        Get group dataframes

        Returns
        -------
        out : tuple or generator
            Group dataframes
        """
        if isinstance(self.data, GroupedDataFrame):
            grouper = self.data.groupby()
            # groupby on categorical columns uses the categories
            # even if they are not present in the data. This
            # leads to empty groups. We exclude them.
            return (gdf for _, gdf in grouper if not gdf.empty)
        else:
            return (self.data, )

    def _evaluate_expressions(self, gdfs):
        """
        Evaluate all group dataframes

        Parameters
        ----------
        gdfs : list
            Dataframes for each group

        Returns
        -------
        out : list
            Result dataframes for each group
        """
        return (self._evaluate_group_dataframe(gdf) for gdf in gdfs)

    def _evaluate_group_dataframe(self, gdf):
        """
        Evaluate a single group dataframe

        Parameters
        ----------
        gdf : pandas.DataFrame
            Input group dataframe

        Returns
        -------
        out : pandas.DataFrame
            Result data
        """
        gdf._is_copy = None
        result_index = gdf.index if self.keep_index else []
        data = pd.DataFrame(index=result_index)
        for expr in self.expressions:
            value = expr.evaluate(gdf, self.env)
            if isinstance(value, pd.DataFrame):
                data = value
                break
            else:
                _create_column(data, expr.column, value)
        data = _add_group_columns(data, gdf)
        return data

    def _concat(self, egdfs):
        """
        Concatenate evaluated group dataframes

        Parameters
        ----------
        egdfs : iterable
            Evaluated dataframes

        Returns
        -------
        edata : pandas.DataFrame
            Evaluated data
        """
        egdfs = list(egdfs)
        edata = pd.concat(egdfs, axis=0, ignore_index=False, copy=False)

        # groupby can mixup the rows. We try to maintain the original
        # order, but we can only do that if the result has a one to
        # one relationship with the original
        one2one = (
            self.keep_index and
            not any(edata.index.duplicated()) and
            len(edata.index) == len(self.data.index))
        if one2one:
            edata = edata.sort_index()
        else:
            edata.reset_index(drop=True, inplace=True)

        # Maybe this should happen in the verb functions
        if self.keep_groups and self.groups:
            edata = GroupedDataFrame(edata, groups=self.groups)
        return edata


class Selector:
    """
    Helper to select columns of a verb
    """
    @staticmethod
    def _resolve_slices(data_columns, names):
        """
        Convert any slices into column names

        Parameters
        ----------
        data_columns : pandas.Index
            Dataframe columns
        names : tuple
            Names (including slices) of columns in the
            dataframe.

        Returns
        -------
        out : tuple
            Names of columns in the dataframe. Has no
            slices.
        """
        def _get_slice_cols(sc):
            """
            Convert slice to list of names
            """
            # Just like pandas.DataFrame.loc the stop
            # column is included
            idx_start = data_columns.get_loc(sc.start)
            idx_stop = data_columns.get_loc(sc.stop) + 1
            return data_columns[idx_start:idx_stop:sc.step]

        result = []
        for col in names:
            if isinstance(col, slice):
                result.extend(_get_slice_cols(col))
            else:
                result.append(col)
        return tuple(result)

    @classmethod
    def verify_columns(cls, selected, data_columns):
        missing_columns = selected.difference(
            data_columns,
            sort=False
        )
        if len(missing_columns):
            raise KeyError(
                "Unknown columns: "
                + ', '.join(missing_columns)
            )
        return selected

    @classmethod
    def has_minus(cls, col, data_columns):
        if isinstance(col, str) and col.startswith('-'):
            return col[1:] in data_columns
        return False

    @classmethod
    def select(cls, verb):
        """
        Return selected columns for the select verb

        Parameters
        ----------
        verb : object
            verb with the column selection attributes:

                - names
                - startswith
                - endswith
                - contains
                - matches
        """
        columns = verb.data.columns
        contains = verb.contains
        matches = verb.matches

        groups = _get_groups(verb)
        names = cls._resolve_slices(columns, verb.names)
        names_set = set(names)
        groups_set = set(groups)
        lst = [[]]

        if names and cls.has_minus(names[0], columns):
            return cls.select_minus(verb)

        if names or groups:
            # group variable missing from the selection are prepended
            missing = [g for g in groups if g not in names_set]
            missing_set = set(missing)
            c1 = missing + [x for x in names if x not in missing_set]
            lst.append(c1)

        if verb.startswith:
            c2 = [x for x in columns
                  if isinstance(x, str) and x.startswith(verb.startswith)]
            lst.append(c2)

        if verb.endswith:
            c3 = [x for x in columns if
                  isinstance(x, str) and x.endswith(verb.endswith)]
            lst.append(c3)

        if contains:
            c4 = []
            for col in columns:
                if (isinstance(col, str) and
                        any(s in col for s in contains)):
                    c4.append(col)
            lst.append(c4)

        if matches:
            c5 = []
            patterns = [x if hasattr(x, 'match') else re.compile(x)
                        for x in matches]
            for col in columns:
                if isinstance(col, str):
                    if any(bool(p.match(col)) for p in patterns):
                        c5.append(col)

            lst.append(c5)

        selected = pd.Index(list(itertools.chain(*lst))).drop_duplicates()

        if verb.drop:
            to_drop = [col for col in selected if col not in groups_set]
            selected = pd.Index(
                [col for col in columns if col not in to_drop]
            )
        return cls.verify_columns(selected, columns)

    @classmethod
    def select_minus(cls, verb):
        columns = verb.data.columns
        names = pd.Index(verb.names).drop_duplicates()
        # Columns preceeded with minus
        exclude_columns = pd.Index([
            col[1:]
            for col in names
            if cls.has_minus(col, columns)
        ])
        # Any other columns
        include_columns = pd.Index([
            col
            for col in names
            if not cls.has_minus(col, columns)
        ])
        selected = columns.difference(exclude_columns, sort=False)

        # For cases like select('-col1', 'col2', 'col1')
        # col1 has to be included
        selected = selected.append(include_columns).drop_duplicates()
        return cls.verify_columns(selected, columns)

    @classmethod
    def _all(cls, verb):
        """
        A verb
        """
        groups = set(_get_groups(verb))
        return [col for col in verb.data if col not in groups]

    @classmethod
    def _at(cls, verb):
        """
        A verb with a select text match
        """
        # Named (listed) columns are always included
        columns = cls.select(verb)
        final_columns_set = set(columns)
        groups_set = set(_get_groups(verb))
        final_columns_set -= groups_set - set(verb.names)
        return [col for col in columns if col in final_columns_set]

    @classmethod
    def _if(cls, verb):
        """
        A verb with a predicate function
        """
        pred = verb.predicate
        data = verb.data
        groups = set(_get_groups(verb))

        # force predicate
        if isinstance(pred, str):
            if not pred.endswith('_dtype'):
                pred = '{}_dtype'.format(pred)
            pred = getattr(pdtypes, pred)
        elif pdtypes.is_bool_dtype(np.array(pred)):
            # Turn boolean array into a predicate function
            it = iter(pred)

            def pred(col):
                return next(it)

        return [col for col in data
                if pred(data[col]) and col not in groups]

    @classmethod
    def get(cls, verb):
        if not hasattr(verb, 'selector') and hasattr(verb, 'names'):
            return cls.select(verb)
        elif verb.selector == '_all':
            return cls._all(verb)
        elif verb.selector == '_at':
            return cls._at(verb)
        elif verb.selector == '_if':
            return cls._if(verb)


def build_expressions(verb):
    """
    Build expressions for helper verbs

    Parameters
    ----------
    verb : verb
        A verb with a *functions* attribute.

    Returns
    -------
    out : tuple
        (List of Expressions, New columns). The expressions and the
        new columns in which the results of those expressions will
        be stored. Even when a result will stored in a column with
        an existing label, that column is still considered new,
        i.e An expression ``x='x+1'``, will create a new_column `x`
        to replace an old column `x`.
    """
    def partial(func, col, *args, **kwargs):
        """
        Make a function that acts on a column in a dataframe

        Parameters
        ----------
        func : callable
            Function
        col : str
            Column
        args : tuple
            Arguments to pass to func
        kwargs : dict
            Keyword arguments to func

        Results
        -------
        new_func : callable
            Function that takes a dataframe, and calls the
            original function on a column in the dataframe.
        """
        def new_func(gdf):
            return func(gdf[col], *args, **kwargs)

        return new_func

    def make_statement(func, col):
        """
        A statement of function called on a column in a dataframe

        Parameters
        ----------
        func : str or callable
            Function to call on a dataframe column
        col : str
            Column
        """
        if isinstance(func, str):
            expr = '{}({})'.format(func, col)
        elif callable(func):
            expr = partial(func, col, *verb.args, **verb.kwargs)
        else:
            raise TypeError("{} is not a function".format(func))
        return expr

    def func_name(func):
        """
        Return name of a function.

        If the function is `np.sin`, we return `sin`.
        """
        if isinstance(func, str):
            return func

        try:
            return func.__name__
        except AttributeError:
            return ''

    # Generate function names. They act as identifiers (postfixed
    # to the original columns) in the new_column names.
    if isinstance(verb.functions, (tuple, list)):
        names = (func_name(func) for func in verb.functions)
        names_and_functions = zip(names, verb.functions)
    else:
        names_and_functions = verb.functions.items()

    # Create statements for the expressions
    # and postfix identifiers
    columns = Selector.get(verb)  # columns to act on
    postfixes = []
    stmts = []
    for name, func in names_and_functions:
        postfixes.append(name)
        for col in columns:
            stmts.append(make_statement(func, col))

    if not stmts:
        stmts = columns

    # Names of the new columns
    # e.g col1_mean, col2_mean, col1_std, col2_std
    add_postfix = (isinstance(verb.functions, dict) or
                   len(verb.functions) > 1)
    if add_postfix:
        fmt = '{}_{}'.format
        new_columns = [fmt(c, p) for p in postfixes for c in columns]
    else:
        new_columns = columns

    expressions = [Expression(stmt, col)
                   for stmt, col in zip(stmts, new_columns)]
    return expressions, new_columns
