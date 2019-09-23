from copy import copy
from collections import defaultdict

import pandas as pd

from .eval import EvalEnvironment
from .types import GroupedDataFrame
from .utils import temporary_attr, custom_dict

# Registry for the verb implementations
# It is of the form {'datatype': {'verbname': verbimplementation}}
REGISTRY = defaultdict(dict)

dataclass_lookup = {
    pd.DataFrame: 'dataframe',
    GroupedDataFrame: 'dataframe',
    custom_dict: 'dict'
}

DATASTORE_TYPES = tuple(dataclass_lookup.keys())


def register_implementations(module, verb_names, datatype):
    """
    Register verb implementations in the module

    Parameters
    ----------
    module : module
        Module with the implementations.
    verb_names : list
        Names of verbs implemented in the module.
    datatype : str
        A name of the datatype implemented. e.g 'dataframe',
        'dict'
    """
    for name in verb_names:
        REGISTRY[datatype][name] = module[name]


# We use this for "single dispatch" instead of maybe
# functools.singledispatch because when piping the datatype
# is not known when the verb class is called.
def get_verb_function(data, verb):
    """
    Return function that implements the verb for given data type
    """
    try:
        datatype = dataclass_lookup[type(data)]
    except KeyError:
        # Some guess work for subclasses
        for klass, type_ in dataclass_lookup.items():
            if isinstance(data, klass):
                datatype = type_
                break
        else:
            raise TypeError(
                "Data of type {} is not supported.".format(type(data))
            )
    try:
        return REGISTRY[datatype][verb]
    except KeyError:
        raise TypeError(
            "Could not find a {} implementation for the verb {} ".format(
                datatype, verb
            )
        )


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
        if len(args) and isinstance(args[0], DATASTORE_TYPES):
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
        # Makes verb reuseable
        self = copy(self)
        self.data = other
        func = get_verb_function(self.data, self.__class__.__name__)
        return func(self)

    def __call__(self, data):
        # This is an undocumented feature, it allows for
        # verb(*args, **kwargs)(df)
        self = copy(self)
        func = get_verb_function(data, self.__class__.__name__)
        with temporary_attr(self, 'data', data):
            result = func(self)
        return result


class DoubleDataDispatch(type):
    """
    Metaclass for single dispatch of double data verbs

    Makes it possible to do::

        data1 >> verb(data2)
        verb(data1, data2)

    and have the same verb work for different types of data
    """
    def __call__(cls, *args, **kwargs):
        # When we have two arguments, we can proceed with
        # the computation
        if len(args) == 2:
            verb = super().__call__(*args, **kwargs)
            func = get_verb_function(verb.x, cls.__name__)
            return func(verb)
        else:
            return super().__call__(*args, **kwargs)


class DoubleDataOperator(metaclass=DoubleDataDispatch):
    """
    Base class for all verbs that operate two dataframes
    """
    def __rrshift__(self, other):
        """
        Overload the >> operator
        """
        verb = copy(self)
        verb.x = other
        func = get_verb_function(verb.x, self.__class__.__name__)
        return func(verb)
