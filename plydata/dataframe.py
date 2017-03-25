import re

import numpy as np
import pandas as pd

from .utils import hasattrs


class verb_methods:
    """
    Verb implementations for a :class:`pandas.DataFrame`
    """
    def mutate(self):
        for col, expr in zip(self.new_columns, self.expressions):
            if isinstance(expr, str):
                value = self.env.eval(expr, inner_namespace=self.data)
            elif len(self.data) == len(value):
                value = expr
            else:
                raise ValueError("Unknown type")
            self.data[col] = value
        return self.data

    def transmute(self):
        d = {}
        for col, expr in zip(self.new_columns, self.expressions):
            if isinstance(expr, str):
                value = self.env.eval(expr, inner_namespace=self.data)
            elif len(self.data) == len(value):
                value = expr
            else:
                raise ValueError("Unknown type")
            d[col] = value

        if d:
            data = pd.DataFrame(d)
        else:
            data = pd.DataFrame(index=self.data.index)

        return data

    def sample_n(self):
        return self.data.sample(**self.kwargs)

    def sample_frac(self):
        return self.data.sample(**self.kwargs)

    def select(self):
        kw = self.kwargs
        columns = self.data.columns
        c0 = np.array([False]*len(columns))
        c1 = c2 = c3 = c4 = c5 = c0

        if self.args:
            c1 = [x in set(self.args) for x in columns]

        if kw['startswith']:
            c2 = [isinstance(x, str) and x.startswith(kw['startswith'])
                  for x in columns]

        if kw['endswith']:
            c3 = [isinstance(x, str) and x.endswith(kw['endswith'])
                  for x in columns]

        if kw['contains']:
            c4 = [isinstance(x, str) and kw['contains'] in x
                  for x in columns]

        if kw['matches']:
            if hasattr(kw['matches'], 'match'):
                pattern = kw['matches']
            else:
                pattern = re.compile(kw['matches'])
            c5 = [isinstance(x, str) and bool(pattern.match(x))
                  for x in columns]

        cond = np.asarray(c1) | c2 | c3 | c4 | c5

        if kw['drop']:
            cond = ~cond

        data = self.data.loc[:, cond]
        if data.is_copy:
            data = data.copy()

        return data

    def rename(self):
        return self.data.rename(columns=self.lookup)

    def distinct(self):
        if hasattrs(self, ('new_columns', 'expressions')):
            verb_methods.mutate(self)
        return self.data.drop_duplicates(subset=self.columns,
                                         keep=self.keep)

    def arrange(self):
        data = self.data
        name_gen = ('col_{}'.format(x) for x in range(100))
        columns = []
        d = {}
        for col, expr in zip(name_gen, self.expressions):
            d[col] = self.env.eval(expr, inner_namespace=self.data)
            columns.append(col)

        if columns:
            df = pd.DataFrame(d).sort_values(by=columns)
            data = data.loc[df.index, :]
            if data.is_copy:
                data = data.copy()

        return data
