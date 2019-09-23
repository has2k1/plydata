import numpy as np

import plydata.dict  # noqa register dict implementation
from plydata import define
from plydata.utils import custom_dict


def test_define():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    d = custom_dict({'x': x})

    # No args
    d >> define()
    assert len(d) == 1

    # All types of args
    result = d >> define(('x*2', 'x*2'),
                         ('x*3', 'x*3'),
                         x_sq='x**2',
                         x_cumsum='np.cumsum(x)',
                         y=y)

    assert len(result) == 6
    assert all(result['x*2'] == x*2)
    assert all(result['x*3'] == x*3)
    assert all(result['x_sq'] == x**2)
    assert all(result['x_cumsum'] == np.cumsum(x))
    assert all(result['y'] == y)
