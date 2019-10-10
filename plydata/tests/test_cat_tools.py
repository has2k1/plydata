import pandas as pd
import pytest
from plydata.cat_tools import (
    cat_anon,
    cat_collapse,
    cat_other,
    cat_reorder2,
    cat_shift,
    cat_shuffle,
)


def test_reorder2():
    c = list('abbccc')
    x = [11, 2, 2, 3, 33, 3]
    y = [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        cat_reorder2(c, x, y+[3])


def test_shuffle():
    c = list('abcde')

    with pytest.raises(TypeError):
        cat_shuffle(c, 'bad_random_state')


def test_shift():
    c = pd.Categorical(list('abcde'))
    res1 = cat_shift(c, len(c))
    res2 = cat_shift(c, len(c)*2)
    res3 = cat_shift(c, -len(c))
    assert res1.equals(res2)
    assert res1.equals(res3)


def test_anon():
    c = list('abcde')

    with pytest.raises(TypeError):
        cat_anon(c, random_state='bad_random_state')


def test_collapse():
    c = pd.Categorical(list('abcdef'), ordered=True)
    mapping = {'first_2': ['a', 'b'], 'second_2': ['e', 'd']}
    result = cat_collapse(c, mapping)
    expected_cats = pd.Index(['first_2', 'c', 'second_2', 'f'])
    assert result.ordered
    assert result.categories.equals(expected_cats)

    mapping = {'other': ['a', 'b'], 'other2': ['c', 'd']}
    result = cat_collapse(c, mapping, group_other=True)
    expected_cats = pd.Index(['other', 'other2', 'other3'])
    assert result.categories.equals(expected_cats)


def test_other():
    c = ['a', 'b', 'a', 'c', 'b', 'b', 'b', 'd', 'c']
    result = cat_other(c, drop='b')
    assert 'b' not in result

    result = cat_other(c, keep='b', other_category='others')
    assert 'b' in result
    assert 'others' in result
    assert 'a' not in result

    with pytest.raises(ValueError):
        cat_other(c, keep='a', drop='b')

    with pytest.raises(ValueError):
        cat_other(c)
