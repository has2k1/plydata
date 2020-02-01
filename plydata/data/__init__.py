import os

import pandas as pd
from pandas.api.types import CategoricalDtype

__all__ = [
    'fish_encounters',
    'gss_cat'
]

__all__ = [str(u) for u in __all__]
_ROOT = os.path.abspath(os.path.dirname(__file__))


def _as_categorical(df, categories, ordered=False):
    """
    Make the columns in df categorical

    Parameters:
    -----------
    categories: dict
        Of the form {str: list},
        where the key the column name and the value is
        the ordered category list
    """
    for col, cats in categories.items():
        df[col] = df[col].astype(CategoricalDtype(cats, ordered=ordered))
    return df


fishdata = pd.read_csv(
    os.path.join(_ROOT, 'fishdata.csv'),
    dtype={'TagID': str}
)
fish_encounters = pd.DataFrame({
    'fish': fishdata['TagID'].astype(
        CategoricalDtype(fishdata['TagID'].unique())
    ),
    'station': fishdata['Station'].astype(
        CategoricalDtype(fishdata['Station'].unique())
    ),
    'seen': fishdata['value']
}).query(
    'seen == 1'
).sort_values(
    'fish'
).reset_index(
    drop=True
)

gss_cat = pd.read_csv(
    os.path.join(_ROOT, 'gss_cat.csv'),
    usecols=range(1, 10)
)
categories = {
    'marital': [
        'No answer', 'Never married', 'Separated', 'Divorced',
        'Widowed', 'Married'
    ],
    'race': [
        'Other', 'Black', 'White', 'Not applicable'
    ],
    'rincome': [
        'No answer', "Don't know", 'Refused', '$25000 or more',
        '$20000 - 24999', '$15000 - 19999', '$10000 - 14999',
        '$8000 to 9999', '$7000 to 7999', '$6000 to 6999', '$5000 to 5999',
        '$4000 to 4999', '$3000 to 3999', '$1000 to 2999', 'Lt $1000',
        'Not applicable'
    ],
    'partyid': [
        'No answer', "Don't know", 'Other party', 'Strong republican',
        'Not str republican', 'Ind,near rep', 'Independent', 'Ind,near dem',
        'Not str democrat', 'Strong democrat'
    ],
    'relig': [
        'No answer', "Don't know", 'Inter-nondenominational',
        'Native american', 'Christian', 'Orthodox-christian',
        'Moslem/islam', 'Other eastern', 'Hinduism', 'Buddhism',
        'Other', 'None', 'Jewish', 'Catholic', 'Protestant',
        'Not applicable'
    ],
    'denom': [
        'No answer', "Don't know", 'No denomination', 'Other', 'Episcopal',
        'Presbyterian-dk wh', 'Presbyterian, merged', 'Other presbyterian',
        'United pres ch in us', 'Presbyterian c in us', 'Lutheran-dk which',
        'Evangelical luth', 'Other lutheran', 'Wi evan luth synod',
        'Lutheran-mo synod', 'Luth ch in america', 'Am lutheran',
        'Methodist-dk which', 'Other methodist', 'United methodist',
        'Afr meth ep zion', 'Afr meth episcopal', 'Baptist-dk which',
        'Other baptists', 'Southern baptist', 'Nat bapt conv usa',
        'Nat bapt conv of am', 'Am bapt ch in usa', 'Am baptist asso',
        'Not applicable'
    ]
}
_as_categorical(gss_cat, categories)

fish_encounters .__doc__ = """
Fish Encounters

.. rubric:: Description

Information about fish swimming down a river: each station represents
an autonomous monitor that records if a tagged fish was seen at that
location. Fish travel in one direction (migrating downstream).
Information about misses is just as important as hits, but is not
directly recorded in this form of the data.

.. rubric:: Format

A dataset with variables:

============   ========================================================
Column         Description
============   ========================================================
fish           Fish identifier
station        Measurement station
seen           Was the fish seen? (1 if yes, and true for all rows)
============   ========================================================

.. rubric:: Source

Dataset provided by Myfanwy Johnston;
more details at
https://fishsciences.github.io/post/visualizing-fish-encounter-histories/
"""


gss_cat.__doc__ = """
A sample of categorical variables from the General Social survey

.. rubric:: Description

A sample of categorical variables from the General Social survey

.. rubric:: Format

============   ===============================================
Column         Description
============   ===============================================
year           Year of survey, 2000–2014
age            Age. Maximum age is truncated to 89 years.
marital        Marital status
race           Race
rincome        Reported income
partyid        Party affiliation
relig          Religion
denom          Denomination
tvhours        Hours per day watching tv
============   ===============================================

.. rubric:: Source

https://gssdataexplorer.norc.org/
"""
