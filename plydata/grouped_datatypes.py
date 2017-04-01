import pandas as pd


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
        if groups:
            self.plydata_groups = list(groups)

    @property
    def _constructor(self):
        return GroupedDataFrame

    def __str__(self):
        s = 'groups: {}\n{}'.format(
            self.plydata_groups,
            super().__str__())
        return s

    def to_html(self, *args, **kwargs):
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
        return group_html + super().to_html(*args, **kwargs)

    def equals(self, other):
        try:
            same_groups = self.plydata_groups == other.plydata_groups
        except AttributeError:
            same_groups = False

        return same_groups and super().equals(other)
