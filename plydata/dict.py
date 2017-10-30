# This module is a stub to show that the dispatch we use
# works for multiple data type sources.

"""
Verb implementations for a :class:`custom_dict`
"""


def define(verb):
    verb.data = verb.data.copy()
    env = verb.env.with_outer_namespace(verb.data)
    for expr in verb.expressions:
        if isinstance(expr.stmt, str):
            value = env.eval(expr.stmt)
        else:
            value = expr.stmt
        verb.data[expr.column] = value
    return verb.data
