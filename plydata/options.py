"""
PlyData Options
"""
# Names of all the options
OPTIONS = {'modify_input_data'}

#: For actions where it may be more efficient, if ``True``
#: the verb modifies the input data. This may be worth it
#: for very large datasets.
#:
#: Examples
#: --------
#: ::
#:
#:     import pandas a pd
#:     from plydata.options import set_option
#:
#:     df = pd.DataFrame({'x': [1, 2, 3]})
#:
#:     df >> define(y='x+1')
#:     'y' in df  # False
#:
#:     set_option('modify_input_data', True)
#:
#:     df >> define(y='x+1')
#:     'y' in df  # True
modify_input_data = False


def get_option(name):
    """
    Get plydata option

    Parameters
    ----------
    name : str
        Name of the option
    """
    if name not in OPTIONS:
        raise ValueError("Unknown option {!r}".format(name))

    return globals()[name]


def set_option(name, value):
    """
    Set plydata option

    Parameters
    ----------
    name : str
        Name of the option
    value : object
        New value of the option

    Returns
    -------
    old : object
        Old value of the option

    See also
    --------
    :class:`options`
    """
    old = get_option(name)
    globals()[name] = value
    return old


class options:
    """
    Options context manager

    The code in the context is run with the specified options.
    This is a convenient wrapper around :func:`set_option` to
    handle setting and unsetting of option values.

    Parameters
    ----------
    kwargs : dict
        ``{option_name: option_value}`` pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> from plydata import define
    >>> from plydata.options import options
    >>> df = pd.DataFrame({'x': [0, 1, 2, 3]})

    With the default options

    >>> df2 = df >> define(y='2*x')
    >>> df2
       x   y
    0  0   0
    1  1   2
    2  2   4
    3  3   6
    >>> df
       x
    0  0
    1  1
    2  2
    3  3

    Using the context manager

    >>> with options(modify_input_data=True):
    ...     df3 = df >> define(z='3*x')
    >>> df3
       x   z
    0  0   0
    1  1   3
    2  2   6
    3  3   9
    >>> df
       x   z
    0  0   0
    1  1   3
    2  2   6
    3  3   9
    >>> df is df3
    True

    The default options apply again.

    >>> df4 = df >> define(w='4*x')
    >>> df
       x   z
    0  0   0
    1  1   3
    2  2   6
    3  3   9
    >>> df is df4
    False
    """
    def __init__(self, **kwargs):
        self.old = {}
        self.new = kwargs

    def __enter__(self):
        for name, value in self.new.items():
            self.old[name] = set_option(name, value)

    def __exit__(self, exc_type, exc_value, traceback):
        for name, value in self.old.items():
            set_option(name, value)
