from .one_table_verbs import *  # noqa
from .two_table_verbs import *  # noqa
from .helper_verbs import *     # noqa
from .expressions import *      # noqa
from .utils import ply          # noqa
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


def _get_all_imports(d):
    """
    Return list of all the imports
    """
    # 1. No local variables
    # 2.`from Module import Something`, puts Module in
    #    the namespace. We do not want that
    import types
    lst = [name for name, obj in d.items()
           if not (name.startswith('_') or
                   isinstance(obj, types.ModuleType))]
    return lst


def _import_and_register_implementations():
    import plydata.dataframe       # noqa
    import plydata.tidy.dataframe  # noqa


_import_and_register_implementations()
__all__ = _get_all_imports(globals())
