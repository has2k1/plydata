Changelog
=========

v0.2.0
------
*(unreleased)*

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
