Changelog
=========
v0.3.3
------
*(2018-08-02)*
- Fixed :class:`~plydata.one_table_verbs.group_indices` for the case
  with no groups.

v0.3.2
------
*(2017-11-27)*

New Features
************
- You can now use slices to :class:`~plydata.one_table_verbs.select`
  columns (:issue:`9`).

v0.3.1
------
*(2017-11-21)*

- Fixed exception with evaluation of grouped categorical columns when
  there are missing categories in the data.

- Fixed issue with ignored groups when
  :class:`~plydata.helper_verbs.count` and
  :class:`~plydata.helper_verbs.add_count` are used with
  a grouped dataframe. The groups list in the verb
  call were ignored.

- Fixed issue where a dataframe with a column named `n`, the column could
  not be referenced (:issue:`6`).

v0.3.0
------
*(2017-11-03)*

- Fixed :class:`~plydata.one_table_verbs.define` (mutate) and
  :class:`~plydata.one_table_verbs.create` (transmute), make them work with
  ``group_by``.

- Fixed :class:`~plydata.helper_verbs.tally` to work with external arrays.

- Fixed :class:`~plydata.helper_verbs.tally` to sort in descending order.

- Fixed the ``nth`` function of :class:`~plydata.one_table_verbs.summarize` to
  return *NaN* when the requested value is out of bounds.

- The ``contains`` and ``matches`` parameters of
  :class:`~plydata.one_table_verbs.select` can now accept a
  :class:`tuple` of values.

- Fixed verbs that create columns (i.e
  :class:`~plydata.one_table_verbs.create`,
  :class:`~plydata.one_table_verbs.define` and
  :class:`~plydata.one_table_verbs.do`)
  so that they can create categorical columns.

- The ``join`` verbs gained *left_on* and *right_on* parameters.

- Fixed verb reuse. You can create a verb and assign it to a variable
  and pipe to the same variable in different operations.

- Fixed issue where :class:`~plydata.one_table_verbs.select` does maintain the
  order in which the columns are listed.

New Features
************

- Added special verb :class:`~plydata.helper_verbs.call`, it allows one to use
  external functions that accept a dataframe as the first argument.

- For :class:`~plydata.one_table_verbs.define`,
  :class:`~plydata.one_table_verbs.create` and
  :class:`~plydata.one_table_verbs.group_by`, you can now use the
  special function ``n()`` to count the number of elements in current
  group.

- Added the single table helper verbs:

    * :class:`~plydata.helper_verbs.add_count`
    * :class:`~plydata.helper_verbs.add_tally`
    * :class:`~plydata.helper_verbs.arrange_all`
    * :class:`~plydata.helper_verbs.arrange_at`
    * :class:`~plydata.helper_verbs.arrange_if`
    * :class:`~plydata.helper_verbs.create_all`
    * :class:`~plydata.helper_verbs.create_at`
    * :class:`~plydata.helper_verbs.create_if`
    * :class:`~plydata.helper_verbs.group_by_all`
    * :class:`~plydata.helper_verbs.group_by_at`
    * :class:`~plydata.helper_verbs.group_by_if`
    * :class:`~plydata.helper_verbs.mutate_all`
    * :class:`~plydata.helper_verbs.mutate_at`
    * :class:`~plydata.helper_verbs.mutate_if`
    * :class:`~plydata.helper_verbs.query_all`
    * :class:`~plydata.helper_verbs.query_at`
    * :class:`~plydata.helper_verbs.query_if`
    * :class:`~plydata.helper_verbs.rename_all`
    * :class:`~plydata.helper_verbs.rename_at`
    * :class:`~plydata.helper_verbs.rename_if`
    * :class:`~plydata.helper_verbs.summarize_all`
    * :class:`~plydata.helper_verbs.summarize_at`
    * :class:`~plydata.helper_verbs.summarize_if`

- Added :class:`~plydata.one_table_verbs.pull` verb.

- Added :class:`~plydata.one_table_verbs.slice_rows` verb.

API Changes
***********
- Using internal function for :class:`~plydata.one_table_verbs.summarize` that
  counts the number of elements in the current group changed from
  ``{n}`` to ``n()``.

- You can now use piping with the two table verbs (the joins).

- ``modify_where`` and ``define_where`` helper verbs have been removed.
  Using the new expression helper functions :class:`~plydata.expressions.case_when`
  and :class:`~plydata.expressions.if_else` is more readable.

- Removed ``dropna`` and ``fillna`` in favour of using
  :class:`~plydata.helper_verbs.call` with :meth:`pandas.DataFrame.dropna` and
  :meth:`pandas.DataFrame.fillna`.


v0.2.1
------
*(2017-09-20)*

- Fixed issue with :class:`~plydata.one_table_verbs.do` and
  :class:`~plydata.one_table_verbs.summarize` where the categorical group columns
  are not categorical in the result.

- Fixed issue with internal modules being imported with
  :py:`from plydata import *`.

- Added :class:`~plydata.one_table_verbs.dropna` and :class:`~plydata.one_table_verbs.fillna`
  verbs. They both wrap around pandas methods of the same name. Now you
  man maintain the pipelining when dealing with most ``NaN`` values.

v0.2.0
------
*(2017-05-06)*

- :class:`~plydata.one_table_verbs.distinct` now uses `pandas.unique` instead of
  :func:`numpy.unique`.

- Added function :func:`~plydata.utils.Q` for quote non-pythonic column
  names in a dataframe.

- Fixed :class:`~plydata.one_table_verbs.query` and :class:`~plydata.one_table_verbs.modify_where`
  query expressions to handle environment variables.

- Added :class:`~plydata.options.options` context manager.

- Fixed bug where some verbs were not reusable. e.g.

  .. code-block:: python

     data = pd.DataFrame({'x': range(5)})
     v = define(y='x*2')
     df >> v  # first use
     df >> v  # Reuse of v

- Added :class:`~plydata.one_table_verbs.define_where` verb, a combination of
  :class:`~plydata.one_table_verbs.define` and :class:`~plydata.one_table_verbs.modify_where`.

v0.1.1
------
*(2017-04-11)*

Re-release of *v0.1.0*

v0.1.0
------
*(2017-04-11)*

First public release
