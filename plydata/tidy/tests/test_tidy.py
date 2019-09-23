import numpy as np
import pandas as pd
import pytest

from plydata import select
from plydata.tidy import (
    gather,
    spread,
    separate,
    pivot_wider
)


# tidy verbs
def test_gather():
    df = pd.DataFrame({
        'name': ['mary', 'oscar', 'martha', 'john'],
        'math': [92, 83, 85, 90],
        'art': [75, 95, 80, 72],
        'pe': [85, 75, 82, 84]
    })
    result = df >> gather('subject', 'grade')
    assert all(result.columns == ['subject', 'grade'])
    assert len(result) == 16

    result1 = df >> gather('subject', 'grade', select('-name'))
    result2 = df >> gather('subject', 'grade', slice('math', 'pe'))
    result3 = df >> gather('subject', 'grade', ['math', 'art', 'pe'])
    result4 = df >> gather('subject', 'grade', '-name')
    assert result2.equals(result1)
    assert result3.equals(result1)
    assert result4.equals(result1)


def test_spread():
    # long form
    df = pd.DataFrame({
        'name': ['mary', 'oscar', 'martha', 'john'],
        'math': [92, 83, 85, 90],
        'art': [75, 95, 80, 72],
        'pe': [85, 75, 82, 84]
    })

    # Test spread-gather round trip
    # df -> gather -> spread -> df
    sdf = df >> gather('subject', 'grade', '-name')
    ldf = sdf >> spread('subject', 'grade')

    # sort and compare
    df1 = df.sort_index(axis=1).sort_values('name').reset_index(drop=True)
    df2 = ldf.sort_index(axis=1).sort_values('name').reset_index(drop=True)
    assert df1.equals(df2)

    # Test convert
    df = pd.DataFrame({
        'variable': np.tile(['a', 'b', 'c', 'd'], 10),
        'types': np.repeat([
            'ints',
            'ints_nans',
            'int_floats',
            'floats',
            'floats_nans',
            'bools',
            'bools_nans',
            'datetime',
            'timedelta',
            'string'
        ], 4),
        'value': [
            1, 2, 3, 4,
            5, 8, np.nan, 8,
            1.0, 2.0, 3.0, 4.0,
            1.3, 2.5, 3.7, 4.9,
            1.3, np.nan, 3.7, 4.9,
            False, True, True, False,
            True, False, True, None,
            *([np.datetime64('2010-01-01'), None]*2),
            *([np.timedelta64('2', 'D'), None]*2),
            'red', 'blue', 'green', 'yellow'
        ]
    })

    result = df >> spread('types', 'value')
    assert result['ints'].dtype == int
    assert result['ints_nans'].dtype == float
    assert result['int_floats'].dtype == float
    assert result['floats'].dtype == float
    assert result['floats_nans'].dtype == float
    assert result['bools'].dtype == bool
    assert result['bools_nans'].dtype == object
    assert result['datetime'].dtype == 'datetime64[ns]'
    assert result['timedelta'].dtype == 'timedelta64[ns]'
    assert result['string'].dtype == object


def test_separate():
    df = pd.DataFrame({
        'alpha': 1,
        'x': ['a,1,1.1,True', 'b,2,2.2,False', 'c,3,3.3,True'],
        'zeta': 2
    })

    result = df >> separate(
        'x', into=['A', 'B', 'C', 'D'], sep=',', convert=True)
    assert len(result.columns) == 6
    assert result['A'].dtype == object
    assert result['B'].dtype == int
    assert result['C'].dtype == float
    assert result['D'].dtype == bool
    assert (result['D'] == [True, False, True]).all()

    # Few separation indices
    with pytest.raises(ValueError):
        df >> separate('x', into=['A', 'B', 'C', 'D'], sep=(1, 3))

    # Additional Pieces & Missing Pieces
    df = pd.DataFrame({
        'alpha': 1,
        'x': ['a,1', 'b', 'c,3,d'],
        'zeta': 6
    })
    with pytest.warns(UserWarning) as record:
        df >> separate('x', into=['A', 'B'])

    messages = [r.message.args[0] for r in record]
    assert len(record) >= 2
    assert any("Additional pieces" in m for m in messages)
    assert any("Missing pieces" in m for m in messages)

    # Missing values
    df = pd.DataFrame({
        'alpha': 1,
        'x': ['a,1', np.nan, 'c,3'],
        'zeta': 6
    })
    result = df >> separate('x', into=['A', 'B'])
    assert np.isnan(result.loc[1, 'A'])
    assert np.isnan(result.loc[1, 'B'])


def test_pivot_wider():
    random_state = np.random.RandomState(123)

    df = pd.DataFrame({
        'name': ['mary', 'oscar'] * 6,
        'face': np.repeat([1, 2, 3, 4, 5, 6], 2),
        'rolls': random_state.randint(5, 21, 12)
    })
    result = df >> pivot_wider(
        names_from='face',
        values_from='rolls',
    )
    assert (result.columns == ['name', 1, 2, 3, 4, 5, 6]).all()

    result = df >> pivot_wider(
        names_from='face',
        values_from='rolls',
        names_prefix='r'
    )
    assert (
        result.columns == ['name', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    ).all()

    # Test convert
    df = pd.DataFrame({
        'variable': np.tile(['a', 'b', 'c', 'd'], 10),
        'types': np.repeat([
            'ints',
            'ints_nans',
            'int_floats',
            'floats',
            'floats_nans',
            'bools',
            'bools_nans',
            'datetime',
            'timedelta',
            'string'
        ], 4),
        'value': [
            1, 2, 3, 4,
            5, 8, np.nan, 8,
            1.0, 2.0, 3.0, 4.0,
            1.3, 2.5, 3.7, 4.9,
            1.3, np.nan, 3.7, 4.9,
            False, True, True, False,
            True, False, True, None,
            *([np.datetime64('2010-01-01'), None]*2),
            *([np.timedelta64('2', 'D'), None]*2),
            'red', 'blue', 'green', 'yellow'
        ]
    })

    result = df >> pivot_wider(
        names_from='types',
        values_from='value',

    )
    assert result['ints'].dtype == int
    assert result['ints_nans'].dtype == float
    assert result['int_floats'].dtype == float
    assert result['floats'].dtype == float
    assert result['floats_nans'].dtype == float
    assert result['bools'].dtype == bool
    assert result['bools_nans'].dtype == object
    assert result['datetime'].dtype == 'datetime64[ns]'
    assert result['timedelta'].dtype == 'timedelta64[ns]'
    assert result['string'].dtype == object
