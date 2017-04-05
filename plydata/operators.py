from importlib import import_module

import pandas as pd

from .eval import EvalEnvironment
from .grouped_datatypes import GroupedDataFrame
from .utils import temporary_attr

type_lookup = {
    pd.DataFrame: import_module('.dataframe', __package__),
    GroupedDataFrame: import_module('.dataframe', __package__),
    dict: import_module('.dict', __package__)
}


# We use this for "single dispatch" instead of maybe
# functools.singledispatch because when piping the datatype
# is not known when the verb class is called.
def get_verb_function(data, verb):
    """
    Return function that implements the verb for given data type
    """
    try:
        module = type_lookup[type(data)]
    except KeyError:
        # Some guess work for subclasses
        for type_, mod in type_lookup.items():
            if isinstance(data, type_):
                module = mod
                break
    try:
        return getattr(module, verb)
    except (NameError, AttributeError):
        msg = "Data source of type '{}' is not supported."
        raise TypeError(msg.format(type(data)))


# Note: An alternate implementation would be to use a decorator
# for the verb classes. The decorator would return a function
# that checks the first argument, logic similar to the one used
# here. The problem with that is that functions are not
# subclassable, and doing the decoration manually at the end of
# the file. The we would have declared classes but exported
# functions, unless we change the names of the classes.
class OptionalSingleDataFrameArgument(type):
    """
    Metaclass for optional dataframe as first argument

    Makes it possible to do both::

        data >> verb(z='x+1')
        verb(data, z='x+1')
    """
    def __call__(cls, *args, **kwargs):
        # When we have data we can proceed with the computation
        if len(args) and isinstance(args[0], pd.DataFrame):
            return args[0] >> super().__call__(*args[1:], **kwargs)
        else:
            return super().__call__(*args, **kwargs)


class DataOperator(metaclass=OptionalSingleDataFrameArgument):
    """
    Base class for all verbs that operate on data
    """
    data = None
    env = None

    def set_env_from_verb_init(self):
        """
        Capture users enviroment

        Should be called from direct subclasses of this class
        """
        # Prevent capturing of wrong environment for when
        # the verb class inherits and calls superclass init
        if not self.env:
            # 0 - local environment
            # 1 - verb init environment
            # 2 - metaclass.__call__ environment
            # 3 - user environment
            self.env = EvalEnvironment.capture(3)

    def __rrshift__(self, other):
        """
        Overload the >> operator
        """
        if self.data is None:
            if isinstance(other, (pd.DataFrame, dict)):
                self.data = other
            else:
                msg = "Unknown type of data {}"
                raise TypeError(msg.format(type(other)))

        func = get_verb_function(self.data, self.__class__.__name__)
        return func(self)

    def __call__(self, data):
        # This is an undocumented feature, it allows for
        # verb(*args, **kwargs)(df)
        func = get_verb_function(data, self.__class__.__name__)
        with temporary_attr(self, 'data', data):
            result = func(self)
        return result


class DoubleDataDispatch(type):
    """
    Metaclass for single dispatch of double data verbs

    Makes it possible to do::

        verb(data1, data2)

    and have the same verb work for different types of data
    """
    def __call__(cls, *args, **kwargs):
        verb = super().__call__(*args, **kwargs)
        func = get_verb_function(args[0], cls.__name__)
        return func(verb)


class DoubleDataOperator(metaclass=DoubleDataDispatch):
    """
    Base class for all verbs that operate two dataframes
    """
