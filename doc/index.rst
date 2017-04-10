.. _index:

plydata
=======
plydata is a library that provides a small grammar for data
manipulation. The grammar consists of verbs that can be applied
to pandas dataframes or database tables. It is based on the R package
`dplyr`_. plydata uses the `>>` operator as a pipe symbol.

At present the only supported data store is the *pandas* dataframe.
We expect to support *sqlite* and maybe *postgresql* and *mysql*.

Example
-------

.. code-block:: python

    import pandas as pd
    from plydata import mutate, query, modify_where

    df = pd.DataFrame({
        'x': [0, 1, 2, 3],
        'y': ['zero', 'one', 'two', 'three']})

    df >> mutate(z='x')
    """
       x      y  z
    0  0   zero  0
    1  1    one  1
    2  2    two  2
    3  3  three  3
    """

    df >> mutate(z=0) >> modify_where('x > 1', z=1)
    """
       x      y  z
    0  0   zero  0
    1  1    one  0
    2  2    two  1
    3  3  three  1
    """

    # You can pass the dataframe as the # first argument
    query(df, 'x > 1')  # same as `df >> query('x > 2')`
    """
       x      y
    2  2    two
    3  3  three
    """


Documentation
-------------

.. toctree::
   :maxdepth: 1

   api
   usage
   installation
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. _dplyr: http://github.com/hadley/dplyr
