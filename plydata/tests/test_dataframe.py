import re

import pandas as pd
import numpy as np
import pytest

from plydata import (mutate, transmute, sample_n, sample_frac, select,
                     rename, distinct, arrange, group_by, summarize)

from plydata.grouped_datatypes import GroupedDataFrame


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

    result = df >> mutate('x*4')
    assert len(result.columns) == 7
    assert all(result['x*4'] == x*4)

    # Branches
    with pytest.raises(ValueError):
        df >> mutate(z=[1, 2, 3, 4])

    with pytest.raises(TypeError):
        df >> mutate(z=object())


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

    result = df >> transmute('x*4')
    assert len(result.columns) == 1
    assert all(result['x*4'] == x*4)

    # Branches
    with pytest.raises(ValueError):
        df >> transmute(z=[1, 2, 3, 4])

    with pytest.raises(TypeError):
        df >> transmute(z=object())


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

    # Branches
    result = df >> arrange()
    assert result is df

    result = df >> arrange('x') >> arrange('y')  # already sorted
    assert result.index.equals(df.index)


def test_group_by():
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
                       'y': [1, 2, 3, 4, 5, 6, 5]})
    result = df >> group_by('x')
    assert isinstance(result, GroupedDataFrame)
    assert result.plydata_groups == ['x']

    result = df >> group_by('x-1', xsq='x**2')
    assert 'x-1' in result
    assert 'xsq' in result
    assert isinstance(result, GroupedDataFrame)


class TestGroupedDataFrame:
    # The verbs should not drop the columns that are grouped on

    df = pd.DataFrame({
        'x': [1, 5, 2, 2, 4, 0, 4],
        'y': [1, 2, 3, 4, 5, 6, 5]
    }) >> group_by('x')

    def test_mutate(self):
        result = self.df.copy() >> mutate(z='2*x')
        assert isinstance(result, GroupedDataFrame)

    def test_transmute(self):
        result = self.df.copy() >> transmute(z='2*x')
        assert 'x' in result
        assert 'z' in result
        assert isinstance(result, GroupedDataFrame)

    def test_sample_n(self):
        result = self.df >> sample_n(5)
        assert 'x' in result
        assert isinstance(result, GroupedDataFrame)

    def test_sample_frac(self):
        result = self.df >> sample_frac(0.25)
        assert 'x' in result
        assert isinstance(result, GroupedDataFrame)

    def test_select(self):
        result = self.df >> select('y')
        assert 'x' in result
        assert isinstance(result, GroupedDataFrame)

    def test_rename(self):
        result = self.df >> rename(y='z')
        assert 'x' in result
        assert 'z' in result
        assert 'y' not in result
        assert isinstance(result, GroupedDataFrame)

    def test_distinct(self):
        result = self.df >> distinct()
        assert isinstance(result, GroupedDataFrame)

    def test_arrange(self):
        result = self.df >> mutate(z='np.sin(x)') >> arrange('z')
        assert isinstance(result, GroupedDataFrame)


def test_summarize():
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
                       'y': [1, 2, 3, 4, 5, 6, 5],
                       'z': [1, 3, 3, 4, 5, 5, 5]})

    result = df >> summarize('np.sum(x)', max='np.max(x)')
    assert result.loc[0, 'max'] == np.max(df['x'])
    assert result.loc[0, 'np.sum(x)'] == np.sum(df['x'])

    result = df >> group_by('y', 'z') >> summarize(mean_x='np.mean(x)')
    assert 'y' in result
    assert 'z' in result
    assert all(result['mean_x'] == [1, 5, 2, 2, 4, 0])

    # Branches
    result = df >> group_by('y') >> summarize('np.sum(z)', constant=1)
    assert 'y' in result
    assert result.loc[0, 'constant'] == 1


class TestAggregateFunctions:
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [0, 0, 1, 1, 2, 3]})

    def test_no_groups(self):
        result = self.df >> summarize('min(x)')
        assert result.loc[0, 'min(x)'] == 0

        result = self.df >> summarize('first(x)')
        assert result.loc[0, 'first(x)'] == 0

        result = self.df >> summarize('last(x)')
        assert result.loc[0, 'last(x)'] == 5

        result = self.df >> summarize('nth(y, 4)')
        assert result.loc[0, 'nth(y, 4)'] == 2

        result = self.df >> summarize('n_distinct(y)')
        assert result.loc[0, 'n_distinct(y)'] == 4

        result = self.df >> summarize('{n}')
        assert result.loc[0, '{n}'] == 6

    def test_groups(self):
        result = self.df >> group_by('y') >> summarize('mean(x)')
        assert all(result['mean(x)'] == [0.5, 2.5, 4, 5])

        result = self.df >> group_by('y') >> summarize('{n}')
        assert all(result['{n}'] == [2, 2, 1, 1])
