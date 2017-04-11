#######
plydata
#######

=========================    =======================
Latest Release               |release|_
License                      |license|_
Build Status                 |buildstatus|_
Coverage                     |coverage|_
Documentation (Dev)          |documentation|_
Documentation (Release)      |documentation_stable|_
=========================    =======================

plydata is a library that provides a grammar for data manipulation.
The grammar consists of verbs that can be applied to pandas
dataframes or database tables. It is based on the R package
`dplyr`_. plydata uses the `>>` operator as a pipe symbol.

At present the only supported data store is the *pandas* dataframe.
We expect to support *sqlite* and maybe *postgresql* and *mysql*.

Installation
============
plydata **only** supports Python 3.

**Official version**

.. code-block:: console

   $ pip install plydata


**Development version**

.. code-block:: console

   $ pip install git+https://github.com/has2k1/plydata.git@master


Example
-------

.. code-block:: python

    import pandas as pd
    from plydata import define, query, modify_where

    df = pd.DataFrame({
        'x': [0, 1, 2, 3],
        'y': ['zero', 'one', 'two', 'three']})

    df >> define(z='x')
    """
       x      y  z
    0  0   zero  0
    1  1    one  1
    2  2    two  2
    3  3  three  3
    """

    df >> define(z=0) >> modify_where('x > 1', z=1)
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

What about dplython or pandas-ply?
----------------------------------

`dplython`_ and `pandas-ply`_ are two other packages that have a similar
objective to plydata. The big difference is plydata does not use
a placeholder variable (`X`) as a stand-in for the dataframe. For example:

.. code-block:: python

    diamonds >> select(X.carat, X.cut, X.price)  # dplython

    diamonds >> select('carat', 'cut', 'price')  # plydata
    select(diamonds, 'carat', 'cut', 'price')    # plydata

For more, see the documentation_.

.. |release| image:: https://img.shields.io/pypi/v/plydata.svg
.. _release: https://pypi.python.org/pypi/plydata

.. |license| image:: https://img.shields.io/pypi/l/plydata.svg
.. _license: https://pypi.python.org/pypi/plydata

.. |buildstatus| image:: https://api.travis-ci.org/has2k1/plydata.svg?branch=master
.. _buildstatus: https://travis-ci.org/has2k1/plydata

.. |coverage| image:: https://coveralls.io/repos/github/has2k1/plydata/badge.svg?branch=master
.. _coverage: https://coveralls.io/github/has2k1/plydata?branch=master

.. |documentation| image:: https://readthedocs.org/projects/plydata/badge/?version=latest
.. _documentation: https://plydata.readthedocs.io/en/latest/

.. |documentation_stable| image:: https://readthedocs.org/projects/plydata/badge/?version=stable
.. _documentation_stable: https://plydata.readthedocs.io/en/stable/

.. _dplyr: http://github.com/hadley/dplyr
.. _pandas-ply: https://github.com/coursera/pandas-ply
.. _dplython: https://github.com/dodger487/dplython

