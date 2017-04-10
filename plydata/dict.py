# This module is a stub to show that the dispatch we use
# works for multiple data type sources.

"""
Verb implementations for a :class:`dict`
"""


def define(verb):
    env = verb.env.with_outer_namespace(verb.data)
    for col, expr in zip(verb.new_columns, verb.expressions):
        if isinstance(expr, str):
            value = env.eval(expr)
        else:
            value = expr
        verb.data[col] = value
    return verb.data
