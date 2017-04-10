import re

import pandas as pd
import numpy as np
import pytest
import numpy.testing as npt

from plydata import (define, create, sample_n, sample_frac, select,
                     rename, distinct, arrange, group_by, ungroup,
                     group_indices, summarize, query, do, head, tail,
                     tally, count, modify_where,
                     inner_join, outer_join, left_join, right_join,
                     anti_join, semi_join)

from plydata.options import set_option
from plydata.grouped_datatypes import GroupedDataFrame


def test_define():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    df = pd.DataFrame({'x': x})

    # No args
    df2 = df >> define()
    assert len(df2.columns) == 1

    # All types of args
    df2 = df >> define(
        ('x*2', 'x*2'),
        ('x*3', 'x*3'),
        x_sq='x**2',
        x_cumsum='np.cumsum(x)',
        y=y,
        w=9)

    assert len(df2.columns) == 7
    assert all(df2['x*2'] == x*2)
    assert all(df2['x*3'] == x*3)
    assert all(df2['x_sq'] == x**2)
    assert all(df2['x_cumsum'] == np.cumsum(x))
    assert all(df2['y'] == y)
    assert all(df2['w'] == 9)

    result = df >> define('x*4')
    assert len(result.columns) == 2

    # Branches
    with pytest.raises(ValueError):
        df >> define(z=[1, 2, 3, 4])


def test_create():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    df = pd.DataFrame({'x': x})

    # No args
    result = df >> create()
    assert len(result.columns) == 0

    # All types of args
    result = df >> create(('x*2', 'x*2'),
                          ('x*3', 'x*3'),
                          x_sq='x**2',
                          x_cumsum='np.cumsum(x)',
                          y=y,
                          w=9)

    assert len(result.columns) == 6
    assert all(result['x*2'] == x*2)
    assert all(result['x*3'] == x*3)
    assert all(result['x_sq'] == x**2)
    assert all(result['x_cumsum'] == np.cumsum(x))
    assert all(result['y'] == y)
    assert all(result['w'] == 9)

    result = df >> create('x*4')
    assert len(result.columns) == 1
    assert all(result['x*4'] == x*4)

    # Branches
    with pytest.raises(ValueError):
        df >> create(z=[1, 2, 3, 4])


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
    result = df >> rename(gong='bell', pin='nail')
    assert len(result.columns) == 4
    assert 'gong' in result.columns
    assert 'pin' in result.columns

    result = df >> rename({'flap': 'tail'}, pin='nail')
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

    result1 = df >> define(z='x%2') >> distinct(['x', 'z'])
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


def test_ungroup():
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
                       'y': [1, 2, 3, 4, 5, 6, 5]})

    result = df >> group_by('x') >> ungroup()
    assert not isinstance(result, GroupedDataFrame)


def test_group_indices():
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
                       'y': [1, 2, 3, 4, 5, 6, 5]})

    results = df >> group_by('x') >> group_indices()
    assert all(results == [1, 4, 2, 2, 3, 0, 3])

    results = df >> group_indices('y % 2')
    assert all(results == [1, 0, 1, 0, 1, 0, 1])


class TestGroupedDataFrame:
    # The verbs should not drop the columns that are grouped on

    df = pd.DataFrame({
        'x': [1, 5, 2, 2, 4, 0, 4],
        'y': [1, 2, 3, 4, 5, 6, 5]
    }) >> group_by('x')

    def test_define(self):
        result = self.df.copy() >> define(z='2*x')
        assert isinstance(result, GroupedDataFrame)

    def test_create(self):
        result = self.df.copy() >> create(z='2*x')
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
        result = self.df >> rename(z='y')
        assert 'x' in result
        assert 'z' in result
        assert 'y' not in result
        assert isinstance(result, GroupedDataFrame)

    def test_distinct(self):
        result = self.df >> distinct()
        assert isinstance(result, GroupedDataFrame)

    def test_arrange(self):
        result = self.df >> define(z='np.sin(x)') >> arrange('z')
        assert isinstance(result, GroupedDataFrame)

    def test_query(self):
        result = self.df >> query('x % 2 == 0')
        assert 'x' in result
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


def test_query():
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [0, 0, 1, 1, 2, 3]})
    result = df >> query('x % 2 == 0')
    assert all(result.loc[:, 'x'] == [0, 2, 4])


def test_do():
    df = pd.DataFrame({'x': [1, 2, 2, 3],
                       'y': [2, 3, 4, 3],
                       'z': list('aabb')})

    def least_squares(gdf):
        X = np.vstack([gdf.x, np.ones(len(gdf))]).T
        (m, c), _, _, _ = np.linalg.lstsq(X, gdf.y)
        return pd.DataFrame({'slope': [m], 'intercept': c})

    def slope(x, y):
        return np.diff(y)[0] / np.diff(x)[0]

    def intercept(x, y):
        return y.values[0] - slope(x, y) * x.values[0]

    def reuse_gdf_func(gdf):
        gdf['c'] = 0
        return gdf

    df1 = df >> group_by('z') >> do(least_squares)
    df2 = df >> group_by('z') >> do(
        slope=lambda gdf: slope(gdf.x, gdf.y),
        intercept=lambda gdf: intercept(gdf.x, gdf.y))

    assert df1.plydata_groups == ['z']
    assert df2.plydata_groups == ['z']

    npt.assert_array_equal(df1['z'],  df2['z'])
    npt.assert_array_almost_equal(df1['intercept'],  df2['intercept'])
    npt.assert_array_almost_equal(df1['slope'],  df2['slope'])

    result = df >> group_by('z') >> do(reuse_gdf_func)
    assert result.loc[0, 'c'] == 0


def test_head():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': list('aaaabbcddd')})

    result = df >> head(2)
    assert len(result) == 2

    result = df >> group_by('y') >> head(2)
    assert len(result) == 7


def test_tail():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': list('aaaabbcddd')})

    result = df >> tail(2)
    assert len(result) == 2

    result = df >> group_by('y') >> tail(2)
    assert len(result) == 7


def test_tally():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'y': ['a', 'b', 'a', 'b', 'a', 'b'],
        'w': [1, 2, 1, 2, 1, 2]})

    result = df >> tally()
    assert result.loc[0, 'n'] == 6

    result = df >> group_by('y') >> tally()
    assert result.loc[:, 'n'].tolist() == [3, 3]

    result = df >> group_by('y') >> tally('w')
    assert result.loc[:, 'n'].tolist() == [3, 6]

    result2 = df >> group_by('y') >> summarize(n='sum(w)')
    assert result.equals(result2)


def test_count():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'y': ['a', 'b', 'a', 'b', 'a', 'b'],
        'w': [1, 2, 1, 2, 1, 2]})

    result = df >> count()
    assert result.loc[0, 'n'] == 6

    result = df >> count('y')
    assert result.loc[:, 'n'].tolist() == [3, 3]

    result = df >> count('y', weights='w')
    assert result.loc[:, 'n'].tolist() == [3, 6]

    result2 = df >> group_by('y') >> summarize(n='sum(w)')
    assert result.equals(result2)


def test_modify_where():
    df = pd.DataFrame({
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 1, 2, 3, 4, 5],
        'z': [0, 1, 2, 3, 4, 5]})

    result = df >> modify_where('x%2 == 0', y='y*10', z=4)
    assert result.loc[[0, 2, 4], 'y'].tolist() == [0, 20, 40]
    assert result.loc[[0, 2, 4], 'z'].tolist() == [4, 4, 4]

    result2 = df >> modify_where('x%2 == 0', ('y', 'y*10'), ('z', 4))
    assert result.equals(result2)

    with pytest.raises(KeyError):
        df >> modify_where('x < 2', w=1)


def test_data_as_first_argument():
    def equals(df1, df2):
        return df1.equals(df2)

    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [0, 0, 1, 1, 2, 3]})

    assert equals(define(df.copy(), 'x*2'), df.copy() >> define('x*2'))
    assert equals(create(df, 'x*2'), df >> create('x*2'))
    assert len(sample_n(df, 5)) == len(df >> sample_n(5))
    assert len(sample_frac(df, .3)) == len(df >> sample_frac(.3))
    assert equals(select(df, 'x'), df >> select('x'))
    assert equals(rename(df.copy(), z='x'), df.copy() >> rename(z='x'))
    assert equals(distinct(df), df >> distinct())
    assert equals(arrange(df, 'np.sin(x)'), df >> arrange('np.sin(x)'))
    assert equals(group_by(df, 'x'), df >> group_by('x'))
    assert equals(ungroup(group_by(df, 'x')),
                  df >> group_by('x') >> ungroup())
    assert equals(summarize(df, 'sum(x)'), df >> summarize('sum(x)'))
    assert equals(query(df, 'x % 2'), df >> query('x % 2'))
    assert equals(tally(df, 'x'), df >> tally('x'))
    assert equals(modify_where(df, 'x<3', y=10),
                  df >> modify_where('x<3', y=10))

    def xsum(gdf):
        return [gdf['x'].sum()]

    assert equals(do(group_by(df, 'y'), xsum=xsum),
                  df >> group_by('y') >> do(xsum=xsum))

    assert len(head(df, 4) == 4)
    assert len(tail(df, 4) == 4)


def test_data_mutability():
    # These tests affirm that we know the consequences of the verbs.
    # A test in the Mutable section should not fail without a change
    # in implementation. That change should be triggered when Pandas
    # implements a consistent copy-on-write policy.
    #
    # When a test in the mutable section fails, it is bad news. The
    # should be no memory usage gains by reusing the original data,
    # except for the case of `rename`.
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [0, 0, 1, 1, 2, 3]})

    # Default to not mutable
    df >> define(z='x**2')
    assert 'z' not in df

    df >> group_by(z='x**2')
    assert 'z' not in df

    df >> modify_where('x<3', y=9)
    assert df.loc[0, 'y'] != 9

    set_option('modify_input_data', True)

    df2 = df.copy()
    df2 >> define(z='x**2')
    assert 'z' in df2

    df2 = df.copy()
    df2 >> group_by(z='x**2')
    assert 'z' in df2

    df2 = df.copy()
    df2 >> modify_where('x<3', y=9)
    assert df2.loc[0, 'y'] == 9

    # Not mutable
    df2 = df.copy()
    df2 >> create(z='x**2')
    assert 'z' not in df2

    df2 >> sample_n(3) >> define(z='x**2')
    assert 'z' not in df2

    df2 >> sample_frac(.5) >> define(z='x**2')
    assert 'z' not in df2

    df2 >> select('x') >> define(z='x**2')
    assert 'z' not in df2

    df2 >> select('x', 'y') >> define(z='x**2')
    assert 'z' not in df2

    # dataframe.rename has copy-on-write (if copy=False) that affects
    # only the new frame. This creates possibility for "action at a
    # distance" effects on the new frame when the original is modified
    result = df2 >> rename(x='z')
    df2['y'] = 3
    result['x'] = 4
    assert 'z' not in df2
    assert df2.loc[0, 'y'] != 4
    assert result.loc[0, 'x'] != 3
    assert result is df2

    df2 >> arrange('x') >> define(z='x**2')
    assert 'z' not in df2

    df2 >> query('x%2') >> define(z='x**2')
    assert 'z' not in df2

    df2 >> group_indices(z='x%2')
    assert 'z' not in df2


def test_joins():
    cols = {'one', 'two', 'three', 'four'}

    df1 = pd.DataFrame({
        'col1': ['one', 'two', 'three'],
        'col2': [1, 2, 3]
    })

    df2 = pd.DataFrame({
        'col1': ['one', 'four', 'three'],
        'col2': [1, 4, 3]
    })

    idf = inner_join(df1, df2, on='col1')
    odf = outer_join(df1, df2, on='col1')
    ldf = left_join(df1, df2, on='col1')
    rdf = right_join(df1, df2, on='col1')
    adf = anti_join(df1, df2, on='col1')
    sdf = semi_join(df1, df2, on='col1')

    # Pandas does all the heavy lifting, simple tests
    # are enough
    assert set(idf['col1']) & cols == {'one', 'three'}
    assert set(odf['col1']) & cols == cols
    assert set(ldf['col1']) & cols == {'one', 'two', 'three'}
    assert set(rdf['col1']) & cols == {'one', 'four', 'three'}
    assert set(adf['col1']) & cols == {'two'}
    assert set(sdf['col1']) & cols == {'one', 'three'}
    assert set(sdf.columns) == {'col1', 'col2'}

    # Preserves group of x frame
    result = inner_join(df1 >> group_by('col1'), df2, on='col1')
    assert isinstance(result, GroupedDataFrame)
    assert result.plydata_groups == ['col1']

    result = inner_join(df2, df1 >> group_by('col1'), on='col1')
    assert not isinstance(result, GroupedDataFrame)
