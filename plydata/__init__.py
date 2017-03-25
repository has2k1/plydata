from .verbs import *  # noqa: F401, F403
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
