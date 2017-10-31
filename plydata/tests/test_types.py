import pandas as pd
from plydata import group_by, ungroup

df = pd.DataFrame({
    'x': [1, 5, 2, 2, 4, 0, 4],
    'y': [1, 2, 3, 4, 5, 6, 5]
}) >> group_by('x')


class TestGroupedDataFrame:
    def test_str(self):
        assert 'groups:' in str(df)

    def test_to_html(self):
        assert '<th>groups</th>' in df.to_html()

    def test_equals(self):
        df1 = df.copy()
        df2 = df >> ungroup()  # Creates DataFrame class

        # Use df.equals since df is a GroupedDataFrame and
        # the groups will be compared
        assert df.equals(df1)
        assert not df.equals(df2)
