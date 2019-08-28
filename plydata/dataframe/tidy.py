"""
Tidy verb initializations
"""
import pandas as pd

from .common import Selector


def gather(verb):
    data = verb.data
    verb._select_verb.data = data
    columns = Selector.get(verb._select_verb)
    exclude = pd.Index(columns).drop_duplicates()
    id_vars = data.columns.difference(exclude, sort=False)
    return pd.melt(data, id_vars, columns, verb.key, verb.value)
