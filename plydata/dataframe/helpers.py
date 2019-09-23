"""
Helper verbs
"""
from functools import wraps

import pandas as pd
import numpy as np

from ..types import GroupedDataFrame
from ..expressions import Expression
from ..operators import register_implementations
from .common import _get_groups, Selector, build_expressions
from .one_table import arrange, create, define, group_by
from .one_table import mutate, rename, summarize

__all__ = ['call', 'tally', 'count', 'add_tally', 'add_count',
           'arrange_all', 'arrange_if', 'arrange_at',
           'create_all', 'create_if', 'create_at',
           'group_by_all', 'group_by_if', 'group_by_at',
           'mutate_all', 'mutate_if', 'mutate_at',
           'query_all', 'query_at', 'query_if',
           'rename_all', 'rename_at', 'rename_if',
           'select_all', 'select_at', 'select_if',
           'summarize_all', 'summarize_if', 'summarize_at']


# Single table verb helpers

def call(verb):
    if isinstance(verb.func, str):
        func = getattr(verb.data, verb.func.lstrip('.'))
        return func(*verb.args, **verb.kwargs)
    else:
        return verb.func(verb.data, *verb.args, **verb.kwargs)


def tally(verb):
    # Prepare for summarize
    if verb.weights is not None:
        if isinstance(verb.weights, str):
            # Summarize will evaluate and sum up the weights
            stmt = 'sum({})'.format(verb.weights)
        else:
            # Do the summation here. The result does not depend
            # on the dataframe.
            stmt = np.sum(verb.weights)
    else:
        stmt = 'n()'

    verb.expressions = [Expression(stmt, 'n')]
    data = summarize(verb)
    if verb.sort:
        data = data.sort_values(by='n', ascending=False)
        data.reset_index(drop=True, inplace=True)

    return data


def count(verb):
    groups = _get_groups(verb)

    # When grouping, add to the any current groups
    verb.add_ = True
    verb.data = group_by(verb)
    data = tally(verb)

    # Restore original groups
    if groups:
        data = GroupedDataFrame(data, groups, copy=False)
    return data


def add_tally(verb):
    if verb.weights:
        if isinstance(verb.weights, str):
            stmt = 'sum({})'.format(verb.weights)
        else:
            stmt = verb.weights
    else:
        stmt = 'n()'

    verb.expressions = [Expression(stmt, 'n')]
    data = define(verb)
    if verb.sort:
        data = data.sort_values(by='n')
        data.reset_index(drop=True, inplace=True)

    return data


def add_count(verb):
    groups = _get_groups(verb)

    # When grouping, add to the any current groups
    verb.add_ = True
    verb.data = group_by(verb)
    data = add_tally(verb)

    if groups:
        # Restore original groups
        data = GroupedDataFrame(data, groups, copy=False)
    else:
        # Remove counted groups
        data = pd.DataFrame(data, copy=False)
    return data


def _query_helper(verb):
    columns = Selector.get(verb)

    # Wrap the predicate in brackets
    fmt = '({})'.format(verb.vars_predicate).format

    # Create tokens expressions for each column
    tokens = [fmt(_=col) for col in columns]

    # Join the tokens into a compound expression
    sep = ' | ' if verb.any_vars else ' & '
    compound_expr = sep.join(tokens)

    # Create final expression, execute and get boolean selectors
    # for the rows
    result_col = '_plydata_dummy_col_'
    verb.expressions = [Expression(compound_expr, result_col)]
    _data = create(verb)
    bool_idx = _data[result_col].values

    data = verb.data.loc[bool_idx, :]
    return data


def _rename_helper(verb):
    # The selector does all the work for the _all, _at & _if
    columns = Selector.get(verb)
    groups = set(_get_groups(verb))
    columns = [c for c in columns if c not in groups]
    func = verb.functions[0]
    new_names = [func(name) for name in columns]
    verb.lookup = {old: new for old, new in zip(columns, new_names)}
    return rename(verb)


def _select_helper(verb):
    # The selector does all the work for the _all, _at & _if
    columns = Selector.get(verb)
    groups = _get_groups(verb)
    groups_set = set(groups)
    columns = [c for c in columns if c not in groups_set]
    func = verb.functions[0]
    new_names = [func(name) for name in columns]
    verb.lookup = {old: new for old, new in zip(columns, new_names)}
    data = rename(verb)
    data = data.loc[:, groups+new_names]
    return data


def _make_verb_helper(verb_func, add_groups=False):
    """
    Create function that prepares verb for the verb function

    The functions created add expressions to be evaluated to
    the verb, then call the core verb function

    Parameters
    ----------
    verb_func : function
        Core verb function. This is the function called after
        expressions created and added to the verb. The core
        function should be one of those that implement verbs that
        evaluate expressions.
    add_groups : bool
        If True, a groups attribute is added to the verb. The
        groups are the columns created after evaluating the
        expressions.

    Returns
    -------
    out : function
        A function that implements a helper verb.
    """

    @wraps(verb_func)
    def _verb_func(verb):
        verb.expressions, new_columns = build_expressions(verb)
        if add_groups:
            verb.groups = new_columns
        return verb_func(verb)

    return _verb_func


arrange_all = _make_verb_helper(arrange)
arrange_if = _make_verb_helper(arrange)
arrange_at = _make_verb_helper(arrange)

create_all = _make_verb_helper(create)
create_if = _make_verb_helper(create)
create_at = _make_verb_helper(create)

group_by_all = _make_verb_helper(group_by, True)
group_by_if = _make_verb_helper(group_by, True)
group_by_at = _make_verb_helper(group_by, True)

mutate_all = _make_verb_helper(mutate)
mutate_if = _make_verb_helper(mutate)
mutate_at = _make_verb_helper(mutate)

query_all = _query_helper
query_at = _query_helper
query_if = _query_helper

rename_all = _rename_helper
rename_at = _rename_helper
rename_if = _rename_helper

select_all = _select_helper
select_at = _select_helper
select_if = _select_helper

summarize_all = _make_verb_helper(summarize)
summarize_if = _make_verb_helper(summarize)
summarize_at = _make_verb_helper(summarize)

register_implementations(globals(), __all__, 'dataframe')
