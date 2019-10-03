import pytest
from plydata.cat_tools import (
    cat_reorder2
)


def test_reorder2():
    c = list('abbccc')
    x = [11, 2, 2, 3, 33, 3]
    y = [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        cat_reorder2(c, x, y+[3])
