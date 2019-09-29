from collections import OrderedDict
import keyword
import re

import pandas as pd
import pandas.api.types as pdtypes
import numpy as np

__all__ = ['case_when', 'if_else']

KEYWORDS = set(keyword.kwlist)

# A pattern that matches the function 'n()'
# anywhere in an expression
n_func_pattern = re.compile(r'\bn\(\)')


# Internal expression classes

class BaseExpression:
    """
    An expression that will be evaluated

    Parameters
    ----------
    stmt : str or function
        Statement that will be evaluated. Some verbs
        allow only one or the other.
    column : str
        Column in which the result of the statment
        will be placed.
    """
    stmt = None
    column = None

    # Whether the statement uses the special function n()
    _has_n_func = False

    def __init__(self, stmt, column):
        self.stmt = stmt
        self.column = column

        # Check for n() in the statement
        if isinstance(stmt, str):
            if n_func_pattern.search(stmt):
                self._has_n_func = True

    def __repr__(self):
        fmt = '{}({!r}, {!r})'.format
        return fmt(self.__class__.__name__, self.stmt, self.column)

    def nice_value(self, value, idx):
        if isinstance(value, (np.ndarray, pd.Series)):
            return value[idx]
        else:
            return value

    def evaluate(self, data, env):
        """
        Evaluate statement

        Parameters
        ----------
        data : pandas.DataFrame
            Data in whose namespace the statement will be
            evaluated. Typically, this is a group dataframe.

        Returns
        -------
        out : object
            Result of the evaluation.pandas.DataFrame
        """
        def n():
            """
            Return number of rows in groups

            This function is part of the public API
            """
            return len(data)

        if isinstance(self.stmt, str):
            # Add function n() that computes the
            # size of the group data to the inner namespace.
            if self._has_n_func:
                namespace = dict(data, n=n)
            else:
                namespace = data
            # Avoid obvious keywords e.g if a column
            # is named class
            if self.stmt not in KEYWORDS:
                value = env.eval(
                    self.stmt,
                    source_name='Expression.evaluate',
                    inner_namespace=namespace)
            else:
                value = namespace[self.stmt]
        elif callable(self.stmt):
            value = self.stmt(data)
        else:
            value = self.stmt
        return value


class CaseWhenExpression(BaseExpression):
    """
    An expression that will be evaluated

    Parameters
    ----------
    preds_values : ordered-dict
        The predicate expressions and value expressions.
        Ordered by most specific to most general.
    column : str
        Column in which the result of the statment
        will be placed.
    """
    def __init__(self, preds_values, column):
        # Both the predicates and values of a case_when are
        # treated as expressions. And they will be evaluated.
        self.pv_expressions = [
            (Expression(pred, None), Expression(value, column))
            for pred, value in preds_values.items()]
        self.preds_values = preds_values
        self.column = column

    def __str__(self):
        fmt = 'CaseWhenExpression({!r}, {!r})'.format
        return fmt(self.preds_values, self.column)

    def evaluate(self, data, env):
        """
        Evaluate the predicates and values
        """
        # For each predicate-value, we keep track of the positions
        # that have been copied to the result, so that the later
        # more general values do not overwrite the previous ones.
        result = np.repeat(None, len(data))
        copied = np.repeat(False, len(data))
        for pred_expr, value_expr in self.pv_expressions:
            bool_idx = pred_expr.evaluate(data, env)
            if not pdtypes.is_bool_dtype(np.asarray(bool_idx)):
                raise TypeError(
                    "The predicate keys must return a boolean array, "
                    "or a boolean value.")
            value = value_expr.evaluate(data, env)
            mask = (copied ^ bool_idx) & bool_idx
            copied |= bool_idx
            idx = np.where(mask)[0]
            result[idx] = self.nice_value(value, idx)
        return np.array(list(result))


class IfElseExpression(BaseExpression):
    def __init__(self, ifelse, column):
        self.stmt = ifelse
        self.column = column
        self.predicate_expr = Expression(ifelse.predicate, None)
        self.true_value_expr = Expression(ifelse.true_value, None)
        self.false_value_expr = Expression(ifelse.false_value, None)

    def evaluate(self, data, env):
        """
        Evaluate the predicates and values
        """
        bool_idx = self.predicate_expr.evaluate(data, env)
        true_value = self.true_value_expr.evaluate(data, env)
        false_value = self.false_value_expr.evaluate(data, env)
        true_idx = np.where(bool_idx)[0]
        false_idx = np.where(~bool_idx)[0]
        result = np.repeat(None, len(data))
        result[true_idx] = self.nice_value(true_value, true_idx)
        result[false_idx] = self.nice_value(false_value, false_idx)
        return np.array(list(result))


def Expression(*args, **kwargs):
    """
    Return an appropriate Expression given the arguments

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the Expression class
    kwargs : dict
        Keyword arguments passed to the Expression class
    """
    # dispatch
    if not hasattr(args[0], '_Expression'):
        return BaseExpression(*args, *kwargs)
    else:
        return args[0]._Expression(*args, **kwargs)


# User API expressions

class case_when(OrderedDict):
    """
    Vectorized case

    Parameters
    ----------
    args : mapping, iterable
        (predicate, value) pairs, ordered from most specific to
        most general.
    kwargs : collections.OrderedDict
        {predicate: value} pairs, ordered from most specific to
        most general.

    Examples
    --------
    >>> import pandas as pd
    >>> from plydata import define
    >>> from plydata.expressions import case_when
    >>> df = pd.DataFrame({'x': range(10)})

    Here we use an iterable of tuples with key-value pairs
    for the predicate and value.

    >>> df >> define(divisible=case_when([
    ...     ('x%2 == 0', 2),
    ...     ('x%3 == 0', 3),
    ...     (True, -1)
    ... ]))
       x  divisible
    0  0          2
    1  1         -1
    2  2          2
    3  3          3
    4  4          2
    5  5         -1
    6  6          2
    7  7         -1
    8  8          2
    9  9          3

    When the most general predicate comes first, it obscures the
    rest. *Every row is matched by atmost one predicate function*

    >>> df >> define(divisible=case_when([
    ...     (True, -1),
    ...     ('x%2 == 0', 2),
    ...     ('x%3 == 0', 3)
    ... ]))
       x  divisible
    0  0         -1
    1  1         -1
    2  2         -1
    3  3         -1
    4  4         -1
    5  5         -1
    6  6         -1
    7  7         -1
    8  8         -1
    9  9         -1

    String values must be quoted

    >>> df >> define(divisible=case_when([
    ...     ('x%2 == 0', '"by-2"'),
    ...     ('x%3 == 0', '"by-3"'),
    ...     (True, '"neither-by-2or3"')
    ... ]))
       x        divisible
    0  0             by-2
    1  1  neither-by-2or3
    2  2             by-2
    3  3             by-3
    4  4             by-2
    5  5  neither-by-2or3
    6  6             by-2
    7  7  neither-by-2or3
    8  8             by-2
    9  9             by-3

    The values can be expressions

    >>> df >> define(divisible=case_when([
    ...     ('x%2 == 0', 'x+200'),
    ...     ('x%3 == 0', 'x+300'),
    ...     (True, -1)
    ... ]))
       x  divisible
    0  0        200
    1  1         -1
    2  2        202
    3  3        303
    4  4        204
    5  5         -1
    6  6        206
    7  7         -1
    8  8        208
    9  9        309

    .. rubric:: Combining Predicates

    When combining predicate statements, you can use the bitwise
    operators, ``|``, ``&``, ``^`` and ``~``. The different
    statements must be enclosed in parenthesis, -- ``()``.

    >>> df >> define(y=case_when([
    ...     ('(x < 5) & (x % 2 == 0)', '"less-than-5-and-even"'),
    ...     ('(x < 5) & (x % 2 != 0)', '"less-than-5-and-odd"'),
    ...     ('(x > 5) & (x % 2 == 0)', '"greater-than-5-and-even"'),
    ...     ('(x > 5) & (x % 2 != 0)', '"greater-than-5-and-odd"'),
    ...     (True, '"Just 5"')
    ... ]))
       x                        y
    0  0     less-than-5-and-even
    1  1      less-than-5-and-odd
    2  2     less-than-5-and-even
    3  3      less-than-5-and-odd
    4  4     less-than-5-and-even
    5  5                   Just 5
    6  6  greater-than-5-and-even
    7  7   greater-than-5-and-odd
    8  8  greater-than-5-and-even
    9  9   greater-than-5-and-odd

    Notes
    -----
    As :class:`dict` classes are ordered, in python 3.6 and above you can
    get away with::

        df >> define(divisible=case_when({
            'x%2 == 0': 'x+200',
            'x%3 == 0': 'x+300',
            True: -1
        }))

    However, be careful it may not always be the case.
    """
    # The expression class that handles user expression
    # of this type.
    _Expression = CaseWhenExpression


class if_else:
    """
    Vectorized if

    Parameters
    ----------
    predicate : bool, str, function
        Predicate
    true_value : object
        Value when predicate is True.
    false_value : object
        Value when predicate is False.

    Examples
    --------
    >>> import pandas as pd
    >>> from plydata import define
    >>> from plydata.expressions import if_else
    >>> df = pd.DataFrame({'x': range(10)})

    y takes on a value that depends on a predicate expression.
    The values can be scalar.

    >>> df >> define(y=if_else('x%2==0', 2, -1))
       x  y
    0  0  2
    1  1 -1
    2  2  2
    3  3 -1
    4  4  2
    5  5 -1
    6  6  2
    7  7 -1
    8  8  2
    9  9 -1

    If they are strings, they should be quoted.

    >>> df >> define(y=if_else('x%2==0', '"even"', '"odd"'))
       x     y
    0  0  even
    1  1   odd
    2  2  even
    3  3   odd
    4  4  even
    5  5   odd
    6  6  even
    7  7   odd
    8  8  even
    9  9   odd

    If the values are treated as expressions.

    >>> df >> define(y=if_else('x%2==0', 'x*2', 'x/2'))
       x     y
    0  0   0.0
    1  1   0.5
    2  2   4.0
    3  3   1.5
    4  4   8.0
    5  5   2.5
    6  6  12.0
    7  7   3.5
    8  8  16.0
    9  9   4.5

    .. rubric:: Combining Predicates

    When combining predicate statements, you can use the bitwise
    operators, ``|``, ``&``, ``^`` and ``~``. The different
    statements must be enclosed in parenthesis, ``()``.

    >>> df >> define(y=if_else(
    ...     '(x < 5) & (x % 2 == 0)',
    ...     '"less-than-5-and-even"',
    ...     '"odd-or-greater-than-5"'))
       x                      y
    0  0   less-than-5-and-even
    1  1  odd-or-greater-than-5
    2  2   less-than-5-and-even
    3  3  odd-or-greater-than-5
    4  4   less-than-5-and-even
    5  5  odd-or-greater-than-5
    6  6  odd-or-greater-than-5
    7  7  odd-or-greater-than-5
    8  8  odd-or-greater-than-5
    9  9  odd-or-greater-than-5
    """
    # The expression class that handles user expression
    # of this type.
    _Expression = IfElseExpression

    def __init__(self, predicate, true_value, false_value):
        self.predicate = predicate
        self.true_value = true_value
        self.false_value = false_value

    def __repr__(self):
        fmt = 'if_else({!r}, {!r}, {!r})'.format
        return fmt(self.predicate, self.true_value, self.false_value)
