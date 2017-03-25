def hasattrs(obj, names):
    """
    Return True of obj has all the names attributes
    """
    return all(hasattr(obj, attr) for attr in names)
