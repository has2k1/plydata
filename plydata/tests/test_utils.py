import pytest

from plydata.utils import hasattrs, temporary_key, temporary_attr, Q


def test_hasattrs():
    class klass:
        pass

    obj = klass()
    obj.a = 'a'
    obj.b = 'b'

    assert hasattrs(obj, ('a',))
    assert hasattrs(obj, ('a', 'b'))
    assert not hasattrs(obj, ('a', 'b', 'c'))


def test_temporary_key():
    d = {'one': 1, 'two': 2}

    with temporary_key(d, 'three', 3):
        assert 'three' in d
    assert 'three' not in d

    # The context does not suppress exceptions
    with pytest.raises(Exception):
        with temporary_key(d, 'four', 4):
            assert 'four' in d
            raise Exception()
    assert 'four' not in d


def test_temporary_attr():
    class klass:
        pass

    obj = klass()

    with temporary_attr(obj, 'one', 1):
        assert obj.one == 1
    assert not hasattr(obj, 'one')

    # The context does not suppress exceptions
    with pytest.raises(Exception):
        with temporary_attr(obj, 'two', 2):
            assert obj.two == 2
            raise Exception()
    assert not hasattr(obj, 'two')


def test_Q():
    a = 1  # noqa: F841
    assert Q("a") == 1
    assert Q("Q") is Q

    with pytest.raises(NameError):
        Q('asdf')
