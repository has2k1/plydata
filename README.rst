#######
PlyData
#######

=================    =================
Latest Release       |release|_
License              |license|_
Build Status         |buildstatus|_
Coverage             |coverage|_
Documentation        |Documentation|_
=================    =================

PlyData makes common data manipulation tasks easy. It provides a small
grammar to manipulate data. It is based on Hadley Wickhams `dplyr`_.
Currently, the only supported data store is the pandas dataframe, but
support for other data stores is not ruled out. We expect to support
`sqlite` in the future. Support for `mysql` and `postgresql` *may* also
be added.


Installation
============

**Official version**

.. code-block:: console

   $ pip install plydata


**Development version**

.. code-block:: console

   $ pip install git+https://github.com/has2k1/plydata.git@master



.. |release| image:: https://img.shields.io/pypi/v/plydata.svg
.. _release: https://pypi.python.org/pypi/plydata

.. |license| image:: https://img.shields.io/pypi/l/plydata.svg
.. _license: https://pypi.python.org/pypi/plydata

.. |buildstatus| image:: https://api.travis-ci.org/has2k1/plydata.svg?branch=master
.. _buildstatus: https://travis-ci.org/has2k1/plydata

.. |coverage| image:: https://coveralls.io/repos/github/has2k1/plydata/badge.svg?branch=master
.. _coverage: https://coveralls.io/github/has2k1/plydata?branch=master

.. |documentation| image:: https://readthedocs.org/projects/plydata/badge/?version=latest
.. _documentation: https://readthedocs.org/projects/plydata/?badge=latest

.. _`dplyr`: http://github.com/hadley/dplyr
