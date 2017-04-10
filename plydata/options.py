"""
PlyData Options
"""

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
    d = globals()

    if name in {'get_option', 'set_option'} or name not in d:
        raise ValueError("Unknown option {}".format(name))

    return d[name]


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
    """
    d = globals()

    if name in {'get_option', 'set_option'} or name not in d:
        raise ValueError("Unknown option {}".format(name))

    old = d[name]
    d[name] = value
    return old
