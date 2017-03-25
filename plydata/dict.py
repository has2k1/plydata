# This module is a stub to show that the dispatch we use
# works for multiple data type sources.


class verb_methods:
    """
    Verb implementations for a :class:`dict`
    """
    def mutate(self):
        env = self.env.with_outer_namespace(self.data)
        for col, expr in zip(self.new_columns, self.expressions):
            if isinstance(expr, str):
                value = env.eval(expr)
            else:
                value = expr
            self.data[col] = value
        return self.data
