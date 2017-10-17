Changelog
=========

v0.3.0
------
*not-yet-released*

- Fixed :class:`~plydata.verbs.define` (mutate) and
  :class:`~plydata.verbs.create` (transmute), make them work with
  ``group_by``.

- Fixed :class:`~plydata.verbs.tally` to work with external arrays.

- Fixed :class:`~plydata.verbs.tally` to sort in descending order.

- Fixed the ``nth`` function of :class:`~plydata.verbs.summarize` to
  return *NaN* when the requested value is out of bounds.

- The ``contains`` and ``matches`` parameters of
  :class:`~plydata.verbs.select` can now accept a
  :class:`tuple` of values.

- Fixed verbs that create columns (i.e
  :class:`~plydata.verbs.create`,
  :class:`~plydata.verbs.define`,
  :class:`~plydata.verbs.define_where` and
  :class:`~plydata.verbs.do`)
  so that they can create categorical columns.

- The ``join`` verbs gained *left_on* and *right_on* parameters.

- Fixed verb reuse. You can create a verb and assign it to a variable
  and pipe to the same variable in different operations.

New Features
************

- Added special verb :class:`~plydata.verbs.call`, it allows one to use
  external functions that accept a dataframe as the first argument.

- :class:`~plydata.verbs.define` You can now use the internal function
  ``n()`` to count the number of elements in current group.

- Added :class:`~plydata.verbs.add_tally` and
  :class:`~plydata.verbs.add_count` verbs.

API Changes
***********
- Using internal function for :class:`~plydata.verbs.summarize` that
  counts the number of elements in the current group changed from
  ``{n}`` to ``n()``.


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
