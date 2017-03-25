import re

import pandas as pd
import numpy as np

from plydata import (mutate, transmute, sample_n, sample_frac, select,
                     rename, distinct, arrange)


def test_mutate():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    df = pd.DataFrame({'x': x})

    # No args
    df >> mutate()
    assert len(df.columns) == 1

    # All types of args
    df >> mutate(('x*2', 'x*2'),
                 ('x*3', 'x*3'),
                 x_sq='x**2',
                 x_cumsum='np.cumsum(x)',
                 y=y)

    assert len(df.columns) == 6
    assert all(df['x*2'] == x*2)
    assert all(df['x*3'] == x*3)
    assert all(df['x_sq'] == x**2)
    assert all(df['x_cumsum'] == np.cumsum(x))
    assert all(df['y'] == y)


def test_transmute():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    df = pd.DataFrame({'x': x})

    # No args
    result = df >> transmute()
    assert len(result.columns) == 0

    # All types of args
    result = df >> transmute(('x*2', 'x*2'),
                             ('x*3', 'x*3'),
                             x_sq='x**2',
                             x_cumsum='np.cumsum(x)',
                             y=y)

    assert len(result.columns) == 5
    assert all(result['x*2'] == x*2)
    assert all(result['x*3'] == x*3)
    assert all(result['x_sq'] == x**2)
    assert all(result['x_cumsum'] == np.cumsum(x))
    assert all(result['y'] == y)


def test_sample_n():
    df = pd.DataFrame({'x': range(20)})
    result = df >> sample_n(10)
    assert len(result) == 10


def test_sample_frac():
    df = pd.DataFrame({'x': range(20)})
    result = df >> sample_frac(0.25)
    assert len(result) == 5


def test_select():
    x = list(range(20))
    df = pd.DataFrame({
        'lion': x, 'tiger': x, 'cheetah': x,
        'leopard': x, 'jaguar': x, 'cougar': x,
        'caracal': x})

    result = df >> select('lion', 'caracal')
    assert len(result.columns) == 2
    assert all(c in result.columns for c in ('lion', 'caracal'))

    result = df >> select(startswith='c')
    assert len(result.columns) == 3

    result = df >> select('caracal', endswith='ar', contains='ee',
                          matches='\w+opa')
    assert len(result.columns) == 5

    # Numerical column names, and regex object
    df[123] = 1
    df[456] = 2
    df[789] = 3
    pattern = re.compile('\w+opa')
    result = df >> select(startswith='t', matches=pattern)
    assert len(result.columns) == 2

    result = df >> select(123, startswith='t', matches=pattern)
    assert len(result.columns) == 3

    result = df >> select(456, 789, drop=True)
    assert len(result.columns) == len(df.columns)-2

    # No selection, should still have an index
    result = df >> select()
    assert len(result.columns) == 0
    assert len(result.index) == len(df.index)


def test_rename():
    x = np.array([1, 2, 3])
    df = pd.DataFrame({'bell': x, 'whistle': x, 'nail': x, 'tail': x})
    result = df >> rename(bell='gong', nail='pin')
    assert len(result.columns) == 4
    assert 'gong' in result.columns
    assert 'pin' in result.columns

    result = df >> rename({'tail': 'flap'}, nail='pin')
    assert len(result.columns) == 4
    assert 'flap' in result.columns
    assert 'pin' in result.columns


def test_distinct():
    # Index                  0, 1, 2, 3, 4, 5, 6
    df = pd.DataFrame({'x': [1, 1, 2, 3, 4, 4, 5],
                       'y': [1, 2, 3, 4, 5, 5, 6]})
    I = pd.Index

    result = df >> distinct()
    assert result.index.equals(I([0, 1, 2, 3, 4, 6]))

    result = df >> distinct(['x'])
    assert result.index.equals(I([0, 2, 3, 4, 6]))

    result = df >> distinct(['x'], 'last')
    assert result.index.equals(I([1, 2, 3, 5, 6]))

    result = df >> distinct(z='x%2')
    assert result.index.equals(I([0, 2]))

    result1 = df >> mutate(z='x%2') >> distinct(['x', 'z'])
    result2 = df >> distinct(['x'], z='x%2')
    assert result1.equals(result2)


def test_arrange():
    # Index                  0, 1, 2, 3, 4, 5
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0],
                       'y': [1, 2, 3, 4, 5, 6]})
    I = pd.Index

    result = df >> arrange('x')
    assert result.index.equals(I([5, 0, 2, 3, 4, 1]))

    result = df >> arrange('x', '-y')
    assert result.index.equals(I([5, 0, 3, 2, 4, 1]))

    result = df >> arrange('np.sin(y)')
    assert result.index.equals(I([4, 3, 5, 2, 0, 1]))
