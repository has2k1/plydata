import re

import pandas as pd
import numpy as np
import pytest
import numpy.testing as npt
import pandas.api.types as pdtypes

from plydata import (define, create, sample_n, sample_frac, select,
                     rename, distinct, arrange, group_by, ungroup,
                     group_indices, summarize, query, do, head, tail,
                     pull, slice_rows,
                     tally, count, add_tally, add_count, call,
                     arrange_all, arrange_at, arrange_if,
                     create_all, create_at, create_if,
                     group_by_all, group_by_at, group_by_if,
                     mutate_all, mutate_at, mutate_if,
                     query_all, query_at, query_if,
                     rename_all, rename_at, rename_if,
                     select_all, select_at, select_if,
                     summarize_all, summarize_at, summarize_if,
                     # Two table verbs
                     inner_join, outer_join, left_join, right_join,
                     anti_join, semi_join,
                     )

from plydata.options import set_option
from plydata.types import GroupedDataFrame
from plydata.helper_verbs import _at as _at_verb


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

    # Works with group_by
    result = df >> group_by('x < 3') >> define(z='len(x)')
    assert all(result['z'] == [2, 2, 1])

    # Potentially problematic index
    def non_range_index_func(s):
        return pd.Series([11, 12, 13], index=[21, 22, 23])

    result = df >> define(z='non_range_index_func(x)')
    assert all(result['z'] == [11, 12, 13])

    # Can create categorical column
    result = df >> define(xcat='pd.Categorical(x)')
    assert all(result['xcat'] == result['x'])
    assert pdtypes.is_categorical_dtype(result['xcat'])

    # Messing with indices
    result = (df
              >> query('x >= 2')
              >> group_by('x')
              >> define(y='x'))
    assert all(result['x'] == result['y'])

    # Do not modify group column
    with pytest.raises(ValueError):
        df >> group_by('x') >> define(x='2*x')

    # Series-like iterables
    # https://github.com/has2k1/plydata/issues/21
    result = df >> define(y=pd.Series(y))
    assert all(result['y'] == y)


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

    # Works with group_by
    result = df >> group_by('x < 3') >> create(z='len(x)')
    assert all(result['z'] == [2, 2, 1])


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
                          matches=r'\w+opa')
    assert len(result.columns) == 5

    result = df >> select(contains=['ee', 'ion', '23'])
    assert len(result.columns) == 2

    result = df >> select(matches=(r'\w+opa', r'\w+r$'))
    assert len(result.columns) == 4

    # grouped on columns are never dropped
    result = df >> group_by('cougar') >> select(startswith='c', drop=True)
    assert len(result.columns) == 5
    assert 'cougar' in result

    # order depends on selection, and grouped columns are prepend
    # if missing from selection
    result1 = df >> select('jaguar', 'lion', 'caracal')
    result2 = df >> select('caracal', 'jaguar', 'lion')
    result3 = df >> group_by('tiger') >> select('caracal', 'jaguar', 'lion')
    assert list(result1.columns) == ['jaguar', 'lion', 'caracal']
    assert list(result2.columns) == ['caracal', 'jaguar', 'lion']
    assert list(result3.columns) == ['tiger', 'caracal', 'jaguar', 'lion']

    # Numerical column names, and regex object
    df[123] = 1
    df[456] = 2
    df[789] = 3
    pattern = re.compile(r'\w+opa')
    result = df >> select(startswith='t', matches=pattern)
    assert len(result.columns) == 2

    result = df >> select(123, startswith='t', matches=pattern)
    assert len(result.columns) == 3

    result = df >> select(456, 789, drop=True)
    assert len(result.columns) == len(df.columns)-2

    result = df >> select(contains=['ee', 'ion'])
    assert len(result.columns) == 2

    # No selection, should still have an index
    result = df >> select()
    assert len(result.columns) == 0
    assert len(result.index) == len(df.index)
    df = pd.DataFrame({
        'lion': x, 'tiger': x, 'cheetah': x,
        'leopard': x, 'jaguar': x, 'cougar': x,
        'caracal': x})

    # Exclude with minus
    result = df >> select('-jaguar', '-lion')
    assert 'jaguar' not in result
    assert 'lion' not in result

    result = df >> select('-jaguar', '-lion', 'jaguar')
    assert result.columns[-1] == 'jaguar'

    # Wrong way to exclude
    with pytest.raises(KeyError):
        df >> select('jaguar', '-lion')

    with pytest.raises(TypeError):
        select.from_columns({})


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
    I = pd.Index  # noqa: E741

    result = df >> distinct()
    assert result.index.equals(I([0, 1, 2, 3, 4, 6]))

    result = df >> distinct(('x', 'y'), z='x+1')
    assert result.index.equals(I([0, 1, 2, 3, 4, 6]))

    result = df >> distinct('last')
    assert result.index.equals(I([0, 1, 2, 3, 5, 6]))

    result = df >> distinct(False)
    assert result.index.equals(I([0, 1, 2, 3, 6]))

    result = df >> distinct(['x'])
    assert result.index.equals(I([0, 2, 3, 4, 6]))

    result = df >> distinct(['x'], 'last')
    assert result.index.equals(I([1, 2, 3, 5, 6]))

    result = df >> distinct(z='x%2')
    assert result.index.equals(I([0, 2]))

    result1 = df >> define(z='x%2') >> distinct(['x', 'z'])
    result2 = df >> distinct(['x'], z='x%2')
    assert result1.equals(result2)

    with pytest.raises(Exception):
        df >> distinct(['x'], 'last', 'cause_exception')


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

    # Bad index
    df_bad = df.copy()
    df_bad.index = [0, 1, 0, 1, 0, 1]
    result = df_bad >> arrange('x')
    assert result.index.equals(I([1, 0, 0, 1, 0, 1]))

    result = df_bad >> arrange('x', '-y')
    assert result.index.equals(I([1, 0, 1, 0, 0, 1]))


def test_group_by():
    df = pd.DataFrame({'x': [1, 5, 2, 2, 4, 0, 4],
                       'y': [1, 2, 3, 4, 5, 6, 5],
                       'z': [1, 2, 3, 4, 5, 6, 5],
                       'w': [1, 2, 3, 4, 5, 6, 5],
                       })
    result = df >> group_by('x')
    assert isinstance(result, GroupedDataFrame)
    assert result.plydata_groups == ['x']

    result = df >> group_by('x-1', xsq='x**2')
    assert 'x-1' in result
    assert 'xsq' in result
    assert isinstance(result, GroupedDataFrame)

    result = df >> group_by('x') >> group_by('y')
    assert result.plydata_groups == ['y']

    result = df >> group_by('x') >> group_by('y', add_=True)
    assert result.plydata_groups == ['x', 'y']

    result = df >> group_by('x', 'w') >> group_by('y', 'x', add_=True)
    assert result.plydata_groups == ['x', 'w', 'y']

    result = df >> group_by(x='2*x', y='2*y')
    assert result.plydata_groups == ['x', 'y']
    npt.assert_array_equal(result['x'], 2*df['x'])


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

    results = df >> group_indices()
    assert all(results == [1, 1, 1, 1, 1, 1, 1])

    # Branches
    with pytest.warns(UserWarning):
        df >> group_by('x') >> group_indices('y')


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

    # (Name, Expression) tuples
    result = df >> summarize(('sum', 'np.sum(x)'), ('max', 'np.max(x)'))
    assert 'sum' in result
    assert 'max' in result

    # Branches
    result = df >> group_by('y') >> summarize('np.sum(z)', constant=1)
    assert 'y' in result
    assert result.loc[0, 'constant'] == 1

    # Category stays category
    df1 = df.copy()
    df1['z'] = pd.Categorical(df1['z'])
    result = df1 >> group_by('y', 'z') >> summarize(mean_x='np.mean(x)')
    assert result['y'].dtype == np.int
    assert pdtypes.is_categorical_dtype(result['z'])


def test_summarize_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> select('x', 'z') >> summarize_all(('mean', np.std))
    expected_cols = ['x_mean', 'z_mean', 'x_std', 'z_std']
    assert len(result.columns.intersection(expected_cols)) == 4
    result.loc[0, 'x_mean'] = 3.5
    result.loc[0, 'z_mean'] = 9.5
    result.loc[0, 'x_std'] = result.loc[0, 'z_std']

    # Group column is not summarized
    result = (df
              >> select('x', 'y', 'z')
              >> group_by('x')
              >> summarize_all(('mean')))
    assert result['x'].equals(df['x'])


def test_summarize_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })
    result = (df
              >> group_by('alpha')
              >> summarize_at(dict(matches=r'x|y'), ('mean', np.std))
              )
    assert all(result.loc[:, 'x_mean'] == [2, 5])
    assert all(result.loc[:, 'y_mean'] == [5, 2])
    assert all(result.loc[:, 'x_std'] == result.loc[:, 'y_std'])


def test_summarize_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    def has4(col):
        return 4 in list(col)

    result = df >> group_by('alpha') >> summarize_if(has4, np.mean)
    assert len(result.columns) == 3
    assert 'alpha' in result
    assert 'x' in result
    assert 'y' in result
    assert 'z' not in result


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

        result = self.df >> summarize('n()')
        assert result.loc[0, 'n()'] == 6

        result = self.df >> summarize(nth='nth(x, 100)')
        assert np.isnan(result.loc[0, 'nth'])

    def test_groups(self):
        result = self.df >> group_by('y') >> summarize('mean(x)')
        assert all(result['mean(x)'] == [0.5, 2.5, 4, 5])

        result = self.df >> group_by('y') >> summarize('n()')
        assert all(result['n()'] == [2, 2, 1, 1])


def test_query():
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [0, 0, 1, 1, 2, 3]})
    c = 3  # noqa: F841

    result = df >> query('x % 2 == 0')
    assert all(result.loc[:, 'x'] == [0, 2, 4])

    result = df >> query('x > @c')
    assert all(result.loc[:, 'x'] == [4, 5])


def test_do():
    df = pd.DataFrame({'x': [1, 2, 2, 3],
                       'y': [2, 3, 4, 3],
                       'z': list('aabb'),
                       'w': pd.Categorical(list('aabb')),
                       })

    def least_squares(gdf):
        X = np.vstack([gdf.x, np.ones(len(gdf))]).T
        (m, c), _, _, _ = np.linalg.lstsq(X, gdf.y, None)
        return pd.DataFrame({'slope': [m], 'intercept': c})

    def slope(x, y):
        return np.diff(y)[0] / np.diff(x)[0]

    def intercept(x, y):
        return y.values[0] - slope(x, y) * x.values[0]

    df1 = df >> group_by('z') >> do(least_squares)
    df2 = df >> group_by('z') >> do(
        slope=lambda gdf: slope(gdf.x, gdf.y),
        intercept=lambda gdf: intercept(gdf.x, gdf.y))

    df3 = df >> group_by('w') >> do(least_squares)
    df4 = df >> group_by('w') >> do(
        slope=lambda gdf: slope(gdf.x, gdf.y),
        intercept=lambda gdf: intercept(gdf.x, gdf.y))

    assert df1.plydata_groups == ['z']
    assert df2.plydata_groups == ['z']
    assert df1['z'].dtype == object
    assert df2['z'].dtype == object
    assert df3['w'].dtype == 'category'
    assert df4['w'].dtype == 'category'

    npt.assert_array_equal(df1['z'],  df2['z'])
    npt.assert_array_almost_equal(df1['intercept'],  df2['intercept'])
    npt.assert_array_almost_equal(df1['slope'],  df2['slope'])

    # No groups (Test with pass-through functions)
    df1 = df >> do(lambda gdf: gdf)
    df2 = df >> do(
        x=lambda gdf: gdf.x,
        y=lambda gdf: gdf.y,
        z=lambda gdf: gdf.z,
        w=lambda gdf: gdf.w
    )

    cols = list('xyzw')
    assert all(df[cols] == df1[cols])
    assert all(df[cols] == df2[cols])

    # Reordered data so that the groups are not all
    # bunched together
    df = pd.DataFrame({'x': [2, 1, 2, 3],
                       'y': [4, 2, 3, 3],
                       'z': list('baab'),
                       'w': pd.Categorical(list('baab')),
                       },
                      index=[3, 1, 0, 2]  # good index
                      )

    dfi = pd.DataFrame({'x': [2, 1, 2, 3],
                        'y': [4, 2, 3, 3],
                        'z': list('baab'),
                        'w': pd.Categorical(list('baab')),
                        },
                       index=[3, 1, 0, 0]  # bad index
                       )

    # Reuse group dataframe
    def sum_x(gdf):
        gdf['sum_x'] = gdf['x'].sum()
        return gdf

    # When the group dataframe is reused and the
    # index is good (no duplicates) the rows
    # in the result should not be reordered
    res = df >> group_by('z') >> do(sum_x)
    assert df['x'].equals(res['x'])
    assert all(res['sum_x'] == [5, 3, 3, 5])

    # Can use string evaluation
    res = df >> group_by('z') >> do(n='len(x)')
    assert all(res['z'] == ['b', 'a'])
    assert all(res['n'] == [2, 2])

    # bad index is handled correctly
    res = dfi >> group_by('z') >> do(sum_x)
    assert dfi.index.equals(res.index)
    assert dfi['x'].equals(res['x'])
    assert all(res['sum_x'] == [5, 3, 3, 5])

    # Branches
    with pytest.raises(ValueError):
        # args and kwargs
        df >> group_by('w') >> do(
            least_squares,
            slope=lambda gdf: slope(gdf.x, gdf.y),
            intercept=lambda gdf: intercept(gdf.x, gdf.y))

    with pytest.raises(TypeError):
        df >> group_by('w') >> do('len(x)')

    # Potentially problematic index
    def non_range_index_func(gdf):
        return pd.Series([11, 12, 13], index=[21, 22, 23])

    result = df >> do(r=non_range_index_func)
    assert all(result['r'] == [11, 12, 13])


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

    # External weights
    result = df >> tally(range(5))
    assert result.loc[0, 'n'] == 10

    # Sort
    result = df >> group_by('y') >> tally('w', sort=True)
    assert result.loc[:, 'n'].tolist() == [6, 3]


def test_count():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'y': ['a', 'b', 'a', 'b', 'a', 'b'],
        'w': [1, 2, 1, 2, 1, 2]})

    result = df >> count()
    assert result.loc[0, 'n'] == 6

    result = df >> count('y')
    assert result.loc[:, 'n'].tolist() == [3, 3]

    result = df >> count('y', 'w')
    assert result.loc[:, 'n'].tolist() == [3, 3]

    result = df >> count('y', weights='w')
    assert result.loc[:, 'n'].tolist() == [3, 6]

    result2 = df >> group_by('y') >> summarize(n='sum(w)')
    assert result.equals(result2)

    result = df >> count('w-1')
    assert result.loc[:, 'w-1'].tolist() == [0, 1]
    assert result.loc[:, 'n'].tolist() == [3, 3]

    result1 = df >> group_by('y') >> count('w')
    result2 = df >> count('y', 'w')
    assert result1.plydata_groups == ['y']
    assert pd.DataFrame(result1).equals(result2)


def test_add_tally():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'y': ['a', 'b', 'a', 'b', 'a', 'b'],
        'w': [1, 2, 1, 2, 1, 2]})
    n = len(df)

    result = df >> add_tally()
    assert all(result['n'] == [6]*n)
    assert not isinstance(result, GroupedDataFrame)

    result = df >> add_tally('w')
    assert all(result['n'] == [9]*n)
    assert not isinstance(result, GroupedDataFrame)

    result = df >> add_tally(11)
    assert all(result['n'] == [11]*n)
    assert not isinstance(result, GroupedDataFrame)

    result = df >> group_by('y') >> add_tally()
    assert all(result['n'] == [3]*n)
    assert isinstance(result, GroupedDataFrame)

    result = df >> group_by('y') >> add_tally('w')
    assert all(result['n'] == [3, 6]*(n//2))
    assert isinstance(result, GroupedDataFrame)

    result1 = df >> group_by('y') >> add_tally('x*w')
    result2 = df >> group_by('y') >> add_tally('x*w', sort=True)
    assert not result2['n'].equals(result1['n'])
    assert result2['n'].tolist() == result1['n'].sort_values().tolist()


def test_add_count():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6],
        'y': ['a', 'b', 'a', 'b', 'a', 'b'],
        'w': [1, 2, 1, 2, 1, 2]})
    n = len(df)

    result = df >> add_count()
    assert all(result['n'] == [6]*n)
    assert not isinstance(result, GroupedDataFrame)

    result = df >> add_count('y')
    assert all(result['n'] == [3]*n)
    assert not isinstance(result, GroupedDataFrame)

    result = df >> group_by('y') >> add_count()
    assert all(result['n'] == [3]*n)
    assert isinstance(result, GroupedDataFrame)

    result1 = df >> add_count('w', 'y')
    result2 = df >> group_by('w') >> add_count('y')
    assert not isinstance(result1, GroupedDataFrame)
    assert result2.plydata_groups == ['w']
    assert pd.DataFrame(result2).equals(result1)


def test_call():
    def remove_column_a(df):
        _df = df.copy()
        del _df['a']
        return _df

    df = pd.DataFrame({'a': [1, 2, 3],
                       'b': [4, 5, np.nan]})

    # External function
    result = df >> call(remove_column_a)
    assert 'a' not in result
    assert 'b' in result

    # dataframe method
    result = df >> call('.dropna')
    assert len(result) == 2

    # dataframe method with arguments
    result = df >> define(c='a*2') >> call('.dropna', axis=1)
    assert 'a' in result
    assert 'b' not in result
    assert 'c' in result


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

    arr = df >> pull('x')
    arr[0] = 99
    assert df.loc[0, 'x'] != 99

    df2 = df >> slice_rows(3)
    df2.loc[0, 'x'] = 999
    assert df.loc[0, 'x'] != 999

    set_option('modify_input_data', True)

    df2 = df.copy()
    df2 >> define(z='x**2')
    assert 'z' in df2

    df2 = df.copy()
    df2 >> group_by(z='x**2')
    assert 'z' in df2

    df2 = df.copy()
    arr = df2 >> pull('x')
    arr[0] = 99
    assert df2.loc[0, 'x'] == 99

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

    set_option('modify_input_data', False)


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

    # Piping
    idf1 = inner_join(df1, df2, on='col1')
    idf2 = df1 >> inner_join(df2, on='col1')
    assert idf1.equals(idf2)

    # Branches
    with pytest.raises(ValueError):
        idf1 = inner_join(df1, df2, 'col1')


def test_Q():
    df = pd.DataFrame({'var.name': [1, 2, 3],
                       'class': [1, 2, 3]})

    with pytest.raises(NameError):
        df >> define(y='var.name')

    with pytest.raises(NameError):
        df >> create(y='var.name')

    with pytest.raises(SyntaxError):
        df >> define(y='class+1')

    with pytest.raises(SyntaxError):
        df >> create(y='class+1')

    with pytest.raises(SyntaxError):
        df >> arrange('class+1')

    df >> define(y='Q("var.name")')
    df >> create(y='Q("var.name")')
    df >> define(y='Q("class")')
    df >> create(y='Q("class")')
    df >> define(y='class')
    df >> create(y='class')
    df >> arrange('class')
    df >> arrange('Q("class")+1')


class TestVerbReuse:
    """
    Test that you can use the same verb for multiple operations
    """
    df1 = pd.DataFrame({'x': [0, 1, 2, 3, 4]})
    df2 = pd.DataFrame({'x': [4, 3, 2, 1, 0]})  # df1 reversed rows

    def _test(self, v):
        """
        Testing method, other methods create verbs
        """
        df1 = self.df1 >> v
        df2 = self.df2 >> v
        df2 = df2[::-1].reset_index(drop=True)  # unreverse
        assert df1.equals(df2)

    def test_define(self):
        v = define(y='x*2')
        self._test(v)

    def test_tally(self):
        v = tally()
        self._test(v)


def test__at_verb():
    with pytest.raises(TypeError):
        _at_verb(object(), np.add)


def test_mutate_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = (df
              >> group_by('alpha')
              >> select('x', 'y', 'z')
              >> mutate_all((np.add, np.subtract), 10)
              )
    assert 'alpha' in result


def test_mutate_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = (df
              >> group_by('alpha')
              >> mutate_at(dict(matches=r'x|y'), np.add, 10))

    assert 'alpha' in result
    assert all(result['x'] == df['x'] + 10)
    assert all(result['y'] == df['y'] + 10)
    assert all(result['z'] == df['z'])
    assert len(result.columns) == len(df.columns)

    # branches
    with pytest.raises(KeyError):
        df >> mutate_at(('x', 'w'), np.add, 10)

    with pytest.raises(TypeError):
        df >> mutate_at(('x', 'y', 'z'), (object(),))

    # Do not modify groups
    with pytest.raises(ValueError):
        df >> group_by('x') >> mutate_at(('x', 'y'), np.sin)


def test_mutate_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    def is_x_or_y(col):
        return col.name in ('x', 'y')

    result = (df
              >> group_by('x')
              >> mutate_if(is_x_or_y, np.add, 10))

    assert all(result['x'] == df['x'])  # Group column is not mutated
    assert all(result['y'] == df['y'] + 10)
    assert len(result.columns) == len(df.columns)


def test_group_by_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> group_by_all()
    assert len(df.columns) == len(result.columns)
    assert len(df.columns) == len(result.plydata_groups)

    result = df >> group_by_all(pd.Categorical)
    assert len(df.columns) == len(result.columns)
    assert len(df.columns) == len(result.plydata_groups)

    result = df >> group_by_all(dict(cat=pd.Categorical))
    assert len(df.columns)*2 == len(result.columns)
    for col in df.columns:
        col_cat = '{}_cat'.format(col)
        assert not pdtypes.is_categorical_dtype(result[col])
        assert pdtypes.is_categorical_dtype(result[col_cat])

    result = (df
              >> group_by('x')
              >> group_by_all(dict(cat=pd.Categorical)))
    assert result.plydata_groups == [
        '{}_cat'.format(col) for col in df.columns if col != 'x']
    assert len(df.columns)*2-1 == len(result.columns)
    assert 'x_cat' not in result


def test_group_by_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> group_by_if('is_integer')
    assert result.plydata_groups == ['x', 'y', 'z']


def test_group_by_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> group_by_at(())
    assert not isinstance(result, GroupedDataFrame)

    result = df >> group_by_at(dict(matches=r'beta|z'))
    assert result.plydata_groups == ['beta', 'z']


def test_create_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> create_all(())
    assert df.equals(result)


def test_create_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> create_at((), ())
    assert len(result) == len(df)
    assert len(result.columns) == 0


def test_create_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> create_if('is_string')
    assert len(result.columns) == 3


def test_arrange_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> select('x', 'y', 'z') >> arrange_all(np.negative)
    assert all(result['x'] == [6, 5, 4, 3, 2, 1])
    assert all(result['y'] == [1, 2, 3, 4, 5, 6])
    assert all(result['z'] == [12, 10, 8, 11, 9, 7])


def test_arrange_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> arrange_if('is_integer', np.negative)
    assert all(result['x'] == [6, 5, 4, 3, 2, 1])
    assert all(result['y'] == [1, 2, 3, 4, 5, 6])
    assert all(result['z'] == [12, 10, 8, 11, 9, 7])


def test_arrange_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result1 = df >> arrange_at(dict(matches=r'.*'))
    result2 = df >> arrange_all()
    assert result1.equals(result2)


def test_select_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> select_all(str.capitalize)
    assert all(result.columns == ['Alpha', 'Beta', 'Theta', 'X', 'Y', 'Z'])

    with pytest.raises(ValueError):
        df >> select_all((str.capitalize, str.upper))


def test_select_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    # order does matter
    result = df >> select_at(('y', 'alpha', 'x'), str.capitalize)
    assert all(result.columns == ['Y', 'Alpha', 'X'])

    # Group is selected and not renamed. Group come before the
    # other columns, which other maintain the listed order.
    result = (df
              >> group_by('beta')
              >> select_at(('x', 'beta', 'alpha'), str.upper))
    assert all(result.columns == ['beta', 'X', 'ALPHA'])

    with pytest.raises(ValueError):
        df >> select_at(('x', 'beta', 'alpha'), (str.capitalize, str.upper))

    with pytest.raises(KeyError):
        df >> select_at(('missing', 'alpha', 'x'), str.capitalize)


def test_select_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> select_if('is_numeric', str.capitalize)
    assert all(result.columns == ['X', 'Y', 'Z'])

    with pytest.raises(ValueError):
        df >> select_if('is_numeric', (str.capitalize, str.upper))


def test_rename_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = df >> group_by('alpha') >> rename_all(str.capitalize)
    assert all(result.columns == ['alpha', 'Beta', 'Theta', 'X', 'Y', 'Z'])

    with pytest.raises(ValueError):
        df >> rename_all((str.capitalize, str.upper))


def test_rename_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    # Group is selected and not renamed
    result = (df
              >> group_by('beta', 'z')
              >> rename_at(('alpha', 'beta', 'x'), str.upper))
    assert all(result.columns == ['ALPHA', 'beta', 'theta', 'X', 'y', 'z'])

    with pytest.raises(ValueError):
        df >> rename_at(('alpha', 'beta', 'x'), (str.capitalize, str.upper))


def test_rename_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    result = (df
              >> group_by('alpha', 'y')
              >> rename_if('is_numeric', str.capitalize))
    assert all(result.columns == ['alpha', 'beta', 'theta', 'X', 'y', 'Z'])

    with pytest.raises(ValueError):
        df >> rename_if('is_numeric', (str.capitalize, str.upper))


def test_query_all():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })
    # Can use a predicate that returns a single boolean value
    # per column.
    result = df >> query_all(any_vars='pdtypes.is_integer_dtype({_})')
    assert result.equals(df)

    # branches
    with pytest.raises(ValueError):
        result = df >> query_all()

    with pytest.raises(ValueError):
        result = df >> query_all(any_vars='sum({_})>4',
                                 all_vars='sum({_})>4')


def test_query_at():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })

    # Where the last value of any column is equal to b
    def last(col):
        return col.iloc[-1]

    result = (df
              >> group_by('alpha')
              >> query_at(dict(endswith='a'), any_vars='last({_}) == "b"'))
    assert pd.DataFrame(result).equals(df.iloc[:3, :])  # alpha = a

    # branches
    with pytest.raises(ValueError):
        result = df >> query_at('is_integer')

    with pytest.raises(ValueError):
        result = df >> query_at('is_integer',
                                any_vars='sum({_})>4',
                                all_vars='sum({_})>4')


def test_query_if():
    df = pd.DataFrame({
        'alpha': list('aaabbb'),
        'beta': list('babruq'),
        'theta': list('cdecde'),
        'x': [1, 2, 3, 4, 5, 6],
        'y': [6, 5, 4, 3, 2, 1],
        'z': [7, 9, 11, 8, 10, 12]
    })
    # This is probably a bad way to do such a comparison.
    # Sum of the z column is not equal to the sum of any other
    # integer column
    result = (df
              >> query_if(
                  'is_integer',
                  any_vars='("{_}" != "x") & (sum({_}) == sum(x))')
              )
    assert len(result) == 6

    # Sum of the x column is equal to the sum at least one other
    # integer column; i.e the y column
    result = (df
              >> query_if(
                  'is_integer',
                  any_vars='("{_}" != "z") & (sum({_}) == sum(z))')
              )
    assert len(result) == 0

    # branches
    with pytest.raises(ValueError):
        result = df >> query_if('is_integer')

    with pytest.raises(ValueError):
        result = df >> query_if('is_integer',
                                any_vars='sum({_})>4',
                                all_vars='sum({_})>4')
