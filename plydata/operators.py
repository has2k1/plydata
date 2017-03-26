from contextlib import suppress
from importlib import import_module
import pandas as pd

from .eval import EvalEnvironment

type_lookup = {
    pd.DataFrame: import_module('.dataframe', __package__),
    dict: import_module('.dict', __package__)
}


def module_for_datatype(data):
    """
    Return module that implements the verbs for given type of data
    """
    with suppress(KeyError):
        return type_lookup[type(data)]

    # Some guess work for subclasses
    for type_, mod in type_lookup.items():
        if isinstance(data, type_):
            return mod

    msg = "Data source of type '{}' is not supported."
    raise TypeError(msg.format(type(data)))


class DataOperator:
    """
    Base class for all verbs that operate on data
    """
    env = None

    def __init__(self, *args, **kwargs):
        # Prevent duplicate captures of the
        # same environment
        if not self.env:
            self.env = EvalEnvironment.capture(2)

    def __rrshift__(self, other):
        """
        Overload the >> operator
        """
        # This method relies on a Python 3 only feature, where
        # we can call an uninstantiated cls.method with any
        # correct no. of parameters. Freedom!
        verb = self.__class__.__name__
        if isinstance(other, DataOperator):
            self.data = other.data
        else:
            self.data = other

        module = module_for_datatype(self.data)
        verb_function = getattr(module, verb)
        return verb_function(self)

    def __call__(self, data):
        # This is an undocumented feature, it allows for
        # verb(*args, **kwargs)(df)
        verb = self.__class__.__name__
        module = module_for_datatype(data)
        verb_function = getattr(module, verb)
        return verb_function(self)
