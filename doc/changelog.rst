Changelog
=========

v0.3.0
------
*not-yet-released*

- Fixed :class:`~plydata.verbs.define` (mutate) and
  :class:`~plydata.verbs.create` (transmute), make them work with
  ``group_by``.

New Features
************

- Added special verb :class:`~plydata.verbs.call`, it allows one to use
  external functions that accept a dataframe as the first argument.

v0.2.1
------
*(2017-09-20)*

- Fixed issue with :class:`~plydata.verbs.do` and
  :class:`~plydata.verbs.summarize` where the categorical group columns
  are not categorical in the result.

- Fixed issue with internal modules being imported with
  :py:`from plydata import *`.

- Added :class:`~plydata.verbs.dropna` and :class:`~plydata.verbs.fillna`
  verbs. They both wrap around pandas methods of the same name. Now you
  man maintain the pipelining when dealing with most ``NaN`` values.

v0.2.0
------
*(2017-05-06)*

- :class:`~plydata.verbs.distinct` now uses `pandas.unique` instead of
  :func:`numpy.unique`.

- Added function :func:`~plydata.utils.Q` for quote non-pythonic column
  names in a dataframe.

- Fixed :class:`~plydata.verbs.query` and :class:`~plydata.verbs.modify_where`
  query expressions to handle environment variables.

- Added :class:`~plydata.options.options` context manager.

- Fixed bug where some verbs were not reusable. e.g.

  .. code-block:: python

     data = pd.DataFrame({'x': range(5)})
     v = define(y='x*2')
     df >> v  # first use
     df >> v  # Reuse of v

- Added :class:`~plydata.verbs.define_where` verb, a combination of
  :class:`~plydata.verbs.define` and :class:`~plydata.verbs.modify_where`.

v0.1.1
------
*(2017-04-11)*

Re-release of *v0.1.0*

v0.1.0
------
*(2017-04-11)*

First public release
