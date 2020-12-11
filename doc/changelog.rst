Changelog
=========

v0.5.0
------
*(not-yet-released)*

Bug Fixes
*********

- Fixed bug in :class:`~plydata.one_table_verbs.arrange` where the sorting would not
  work in some cases when the dataframe index was out of order.

API Changes
***********

- :class:`~plydata.one_table_verbs.arrange`,
 :class:`~plydata.one_table_verbs.query`,
 :class:`~plydata.helpers.query_all`,
 :class:`~plydata.helpers.query_at`,
 :class:`~plydata.helpers.query_if`,
 :class:`~plydata.helpers.arrange_all`,
 :class:`~plydata.helpers.arrange_at` and
 :class:`~plydata.helpers.arrange_if` now return dataframe with the indices reset.

v0.4.3
------
*(2020-12-08)*

Bug Fixes
*********

- This release makes Plydata depend on pandas >= 1.1.5.

v0.4.2
------
*(2020-09-12)*

- This is release makes Plydata depend on pandas < 1.1.0. See
  `Issue 23 <https://github.com/has2k1/plydata/issues/23>`_ for details.


v0.4.1
------
*(2020-06-10)*

Bug Fixes
*********

- Fixed bug in :class:`~plydata.one_table_verbs.define` where you could not
  create a new column from array-like or series-like iterables. (:issue:`21`)

- Fixed bug in :class:`~plydata.one_table_verbs.arrange` where dataframes with
  irregular indicies would give wrong output. (:issue:`22`)

v0.4.0
------
*(2020-03-15)*

Bug Fixes
*********

- :class:`~plydata.one_table_verbs.query` now works within groups.

New Features
************
- Added :class:`~plydata.tidy_verbs.gather` to transform dataframe from
  wide-form to long-form.

- Added :class:`~plydata.tidy_verbs.spread` to transform dataframe from
  long-form to wide-form

- Added :class:`~plydata.tidy_verbs.separate` to split a string variable/
  column into different variables/columns.

- Added :class:`~plydata.tidy_verbs.extract` which uses a regular expression
  with groups to extract one or more variables different columns.

- Added :class:`~plydata.tidy_verbs.pivot_wider` to transform dataframe from
  long-form to wide-form. This is a more general version of
  :class:`~plydata.tidy_verbs.spread`.

- Added :class:`~plydata.tidy_verbs.pivot_longer` to transform dataframe from
  wide-form to long-form. This is a more general version of
  :class:`~plydata.tidy_verbs.gather`.

- Added :class:`~plydata.tidy_verbs.separate_rows` to split multiple delimited
  values and place each one in its own row.

- Added :class:`~plydata.tidy_verbs.unite` to join multiple columns into one.

- Added :class:`~plydata.cat_tools.cat_inorder` which creates a categorical
  with categories *in order* of how they appear in the sequence.

- Added :class:`~plydata.cat_tools.cat_infreq` which creates a categorical
  with categories in order of the number of times they appear in the sequence.

- Added :class:`~plydata.cat_tools.cat_inseq` which creates a categorical
  with categories in ascending numerical order.

- Added :class:`~plydata.cat_tools.cat_reorder` which creates a categorical
  with categories ordered according to another variable.

- Added :class:`~plydata.cat_tools.cat_reorder2` which creates a categorical
  with categories ordered according a relationship between two other variables.

- Added :class:`~plydata.cat_tools.cat_rev` which creates a categorical
  with reversed categories.

- Added :class:`~plydata.cat_tools.cat_shuffle` which creates a categorical
  with the categories in a random order.

- Added :class:`~plydata.cat_tools.cat_shift` which creates a categorical
  with the categories shifted to the left or to the right.

- Added :class:`~plydata.cat_tools.cat_move`
  (:class:`~plydata.cat_tools.cat_relevel`) which creates a categorical
  with the categories moved to a given position.

- Added :class:`~plydata.cat_tools.cat_anon` which creates a categorical
  with the categories renamed and reordered with arbitrary numeric identifiers.

- Added :class:`~plydata.cat_tools.cat_collapse` which creates a categorical
  with new umbrella categories that combine one or more of the original
  categories.

- Added :class:`~plydata.cat_tools.cat_other` which creates a categorical
  with a new umbrella category that combines one or more of the original
  categories.

- Added :class:`~plydata.cat_tools.cat_lump` which lumps together most/least
  common categories.

- Added :class:`~plydata.cat_tools.cat_lump_min` which lumps together common
  enough categories.

- Added :class:`~plydata.cat_tools.cat_rename` with which you can manually
  change category names (and values).

- Added :class:`~plydata.cat_tools.cat_relabel` to change category names
  using a function.

- Added :class:`~plydata.cat_tools.cat_expand` to add or remove categories to
  a categorical.

- Added :class:`~plydata.cat_tools.cat_explicit_na` to create a category for
  missing values.

- Added :class:`~plydata.cat_tools.cat_remove_unsed` to remove/drop unused
  categories.

- Added :class:`~plydata.cat_tools.cat_unify` to unify (union of all) the
  categories in a list of categoricals.

- Added :class:`~plydata.cat_tools.cat_concat` to concantenate categoricals
  and combine the categories.

- Added :class:`~plydata.cat_tools.cat_zip` to combine two or more categoricals.

- Added :class:`~plydata.utils.ply` function. Makes it possible to use
  plydata with implied piping without abusing the ``>>`` operator.
  It is also more efficient as it minimises the copying of data.

- Added :class:`~plydata.cat_tools.cat_lump_n`,
  :class:`~plydata.cat_tools.cat_lump_prop`, and
  :class:`~plydata.cat_tools.cat_lump_lowfreq` as the distinct cases of
  :class:`~plydata.cat_tools.cat_lump`.


Enhancements
************

- You cannot modify variables that have been grouped on, an exception is
  raised.

.. code-block:: python

    df = pd.DataFrame({'x': [1, 1, 2], 'y': [1, 2, 3]])})
    df >> define(x='2*x')                   # Correct
    df >> group_by('x') >> define(x='2*x')  # Error

- Fixed :class:`~plydata.one_table_verbs.select` can now exclude columns
  that are prepend with a ``-``

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
