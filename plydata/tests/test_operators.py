import pandas as pd
import pytest
from plydata import define

from plydata.operators import get_verb_function


def test_get_verb_function():
    class dataframe_subclass(pd.DataFrame):
        pass

    data = pd.DataFrame()
    data2 = dataframe_subclass()

    # No error
    func1 = get_verb_function(data, 'define')
    func2 = get_verb_function(data2, 'define')
    assert func1 is func2

    with pytest.raises(TypeError):
        get_verb_function(data, 'unknown_verb')


def test_DataOperator():
    s = {1, 2, 3}  # unrecognized datastore
    data = pd.DataFrame({'x': [1, 2, 3],
                         'y': [1, 2, 3]})

    with pytest.raises(TypeError):
        s >> define(z='x')

    # Currying
    result = define(z=[3, 2, 1])(data)
    assert 'x' in result
    assert 'y' in result
    assert 'z' in result
