import pandas as pd
import pytest
from plydata.cat_tools import (
    cat_anon,
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
