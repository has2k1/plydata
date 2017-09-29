import pandas as pd
import pytest

from plydata import define
from plydata.options import options, set_option, get_option


def test_options_context():
    # Straight test
    set_option('modify_input_data', False)
    assert not get_option('modify_input_data')
    with options(modify_input_data=True):
        assert get_option('modify_input_data')
    assert not get_option('modify_input_data')

    # With some data
    df = pd.DataFrame({'x': [0, 1, 2, 3]})

    df2 = df >> define(y='2*x')
    assert not df.equals(df2)

    with options(modify_input_data=True):
        df3 = df >> define(z='3*x')
    assert df.equals(df3)
    assert df is df3

    df4 = df >> define(w='4*x')
    assert not df.equals(df4)

    # That the options context manager should not muffle
    # an exception.
    with pytest.raises(ValueError):
        with options(modify_input_data=True):
            raise ValueError()

    # The above exception should not leave a modified option
    assert not get_option('modify_input_data')

    with pytest.raises(ValueError):
        assert not get_option('time_travel')
