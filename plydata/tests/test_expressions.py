from collections import OrderedDict
import pandas as pd
import pytest

from plydata.expressions import BaseExpression, CaseWhenExpression
from plydata.expressions import IfElseExpression, if_else
from plydata.utils import get_empty_env


def test_BaseExpression():
    df = pd.DataFrame({'x': [1, 2, 3, 4]})

    env = get_empty_env()
    env = env.with_outer_namespace({'w': 3})

    expr = BaseExpression('sum(x)', 'x_sum')
    assert str(expr) == "BaseExpression('sum(x)', 'x_sum')"
    assert expr.evaluate(df, env) == 10

    expr = BaseExpression('sum(x*w)', 'x_weighted_sum')
    assert str(expr) == "BaseExpression('sum(x*w)', 'x_weighted_sum')"
    assert expr.evaluate(df, env) == 30


def test_CaseWhenExpression():
    df = pd.DataFrame({'x': [1, 2, 3, 4]})

    env = get_empty_env()
    expr = CaseWhenExpression(OrderedDict([
        ('x <= 2', 20),
        ('x <= 3', 30),
        (True, 99)
    ]), 'y')
    value = expr.evaluate(df, env)
    assert all(value == [20, 20, 30, 99])

    # Branches
    expr = CaseWhenExpression(OrderedDict([
        ('x + 2', 20),
        ('x < 3', 30),
        (True, 99)
    ]), 'y')
    with pytest.raises(TypeError):
        expr.evaluate(df, env)
    expr = CaseWhenExpression(
        OrderedDict([('x%2==0', 0), ('x%2==1', 1)]), 'y')
    assert str(expr) == (
        "CaseWhenExpression("
        "OrderedDict([('x%2==0', 0), ('x%2==1', 1)]), 'y')")


def test_IfElseExpression():
    expr = IfElseExpression(if_else('x%2==0', 2, -1), 'y')
    assert str(expr) == \
        "IfElseExpression(if_else('x%2==0', 2, -1), 'y')"


def test_n():
    df = pd.DataFrame({'n': [1, 2, 3, 4],
                       'x': [1, 1, 2, 2]})
    env = get_empty_env()

    expr = BaseExpression('n', 'n')
    value = expr.evaluate(df, env)
    assert all(value == df['n'])

    expr = BaseExpression('n()', 'n')
    value = expr.evaluate(df, env)
    assert value == len(df)

    expr = BaseExpression('x/n()', 'x_ratio')
    value = expr.evaluate(df, env)
    assert all(value == df['x']/len(df))

    with pytest.raises(TypeError):
        expr = BaseExpression('n/n()', 'n_ratio')
        value = expr.evaluate(df, env)
