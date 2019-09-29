import pandas as pd
import numpy as np


class GroupedDataFrame(pd.DataFrame):
    """
    Grouped DataFrame

    This is just a :class:`pandas.DataFrame` with information
    on how to do the grouping.
    """
    # See: subclassing-pandas-data-structures at
    # http://pandas.pydata.org/pandas-docs/stable/internals.html
    _metadata = ['plydata_groups']

    plydata_groups = None

    def __init__(self, data=None, groups=None, **kwargs):
        super().__init__(data=data, **kwargs)
        if groups is not None:
            self.plydata_groups = list(pd.unique(groups))

    @property
    def _constructor(self):
        return GroupedDataFrame

    def __repr__(self):
        s = 'groups: {}\n{}'.format(
            self.plydata_groups,
            super().__repr__())
        return s

    def groupby(self, by=None, **kwargs):
        """
        Group by and do not sort (unless specified)

        For plydata use cases, there is no need to specify
        group columns.
        """
        if by is None:
            by = self.plydata_groups

        # Turn off sorting by groups messes with some verbs
        if 'sort' not in kwargs:
            kwargs['sort'] = False

        return super().groupby(by, **kwargs)

    def group_indices(self):
        """
        Return group indices
        """
        # No groups
        if not self.plydata_groups:
            return np.ones(len(self), dtype=int)

        grouper = self.groupby()
        indices = np.empty(len(self), dtype=int)
        for i, (_, idx) in enumerate(sorted(grouper.indices.items())):
            indices[idx] = i
        return indices

    def _repr_html_(self, *args, **kwargs):
        cell = '<td>{}</td>'
        td_gen = (cell.format(x) for x in self.plydata_groups)
        group_html = """
        <table border="1" class="dataframe">
            <tbody>
                <tr>
                    <th>groups</th>
                    {}
                </tr>
            </tbody>
        </table>
        """.format(''.join(td_gen))
        return group_html + super()._repr_html_(*args, **kwargs)

    def to_html(self, *args, **kwargs):
        return self._repr_html_(*args, **kwargs)

    def equals(self, other):
        try:
            same_groups = self.plydata_groups == other.plydata_groups
        except AttributeError:
            same_groups = False

        return same_groups and super().equals(other)
