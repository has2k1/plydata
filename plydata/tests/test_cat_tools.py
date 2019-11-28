import pandas as pd
import pytest
from plydata.cat_tools import (
    cat_anon,
    cat_collapse,
    cat_explicit_na,
    cat_lump,
    cat_lump_min,
    cat_other,
    cat_remove_unused,
    cat_rename,
    cat_reorder2,
    cat_shift,
    cat_shuffle,
    cat_unify,
    cat_zip,
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


def test_lump():
    result = cat_lump([])
    expected_cats = pd.Index([])
    assert result.categories.equals(expected_cats)

    result = cat_lump(['a'])
    expected_cats = pd.Index(['a'])
    assert result.categories.equals(expected_cats)

    result = cat_lump(['a'], n=1)
    expected_cats = pd.Index(['a'])
    assert result.categories.equals(expected_cats)

    result = cat_lump(['a'], prop=0.5)
    expected_cats = pd.Index(['a'])
    assert result.categories.equals(expected_cats)

    result = cat_lump(['a'], prop=-0.5)
    expected_cats = pd.Index(['other'])
    assert result.categories.equals(expected_cats)

    # No lumping since no category has is more than 0.5
    # (50%) of the values
    result = cat_lump(['a', 'b', 'c'], prop=-0.5)
    expected_cats = pd.Index(['a', 'b', 'c'])
    assert result.categories.equals(expected_cats)


def test_lump_min():
    result = cat_lump_min([], min=1)
    expected_cats = pd.Index([])
    assert result.categories.equals(expected_cats)

    c = pd.Categorical(list('abccdd'), categories=list('dcba'))
    result = cat_lump_min(c, min=2)
    expected_cats = pd.Index(['d', 'c', 'other'])
    assert result.categories.equals(expected_cats)


def test_rename():
    c = pd.Categorical([], categories=list('bacd'))
    result = cat_rename(c, b='B', d='D')
    expected_cats = pd.Index(list('BacD'))
    assert result.categories.equals(expected_cats)

    c = list('abcd')
    result = cat_rename(c)
    expected_cats = pd.Index(list('abcd'))
    assert result.categories.equals(expected_cats)

    with pytest.raises(ValueError):
        cat_rename(c, mapping={'a': 'A'}, b='B', d='D')

    with pytest.raises(IndexError):
        cat_rename(c, z='Z')


def test_explict_na():
    result = cat_explicit_na(['a', None, 'b'])
    expected_cats = pd.Index(['a', 'b', '(missing)'])
    assert result.categories.equals(expected_cats)


def test_drop():
    result = cat_remove_unused(list('abcd'))
    expected_cats = pd.Index(list('abcd'))
    assert result.categories.equals(expected_cats)


def test_unify():
    c1 = list('ab')
    c2 = list('dac')
    result = cat_unify([c1, c2], list('xw'))
    expected_cats = pd.Index(list('abcdxw'))
    assert result[0].categories.equals(expected_cats)
    assert result[1].categories.equals(expected_cats)


def test_zip():
    c1 = pd.Categorical(list('ab'), list('ba'))
    c2 = pd.Categorical(list('12'), list('12'))

    result = cat_zip(c1, c2, keep_empty=False)
    expected_cats = pd.Index(['b:2', 'a:1'])
    assert result.categories.equals(expected_cats)

    result = cat_zip(c1, c2, keep_empty=True)
    expected_cats = pd.Index(['b:1', 'b:2', 'a:1', 'a:2'])
    assert result.categories.equals(expected_cats)
