Usage
=====

For data manipulation, there are two types of verbs;

1. :ref:`api:One table verbs`
2. :ref:`api:Two table verbs`

We define the usage by how the verbs accept the data argument.
There are three types of usage;

1. Piping - The data is to the left side of the pipe symbol (``>>``).
2. Composing - The data is the first argument of the verb.
3. Currying - The data is the only argument of an instantiated verb.

The single table verbs support all three types, while two table verbs
only support composing. Here is an example of the same operation in
three different ways.

.. code-block:: python

   df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5]})

   # Piping
   df >> define(w='x%2', y='x+1', z='x+2.5') >> arrange('w')

   # Composing
   arrange(define(df, w='x%2', y='x+1', z='x+2.5'), 'w')

   # Currying
   arrange('w')(define(w='x%2', y='x+1', z='x+2.5')(df))

Although *composing* is the normal way calls are invoked, since data
manipulation often involves consecutive function calls, the
nested invocations become hard to read. *Piping* helps improve
readability. *Currying* exists only to spite the `zen of python`_.

Data mutability
---------------

By default, plydata does not modify in input dataframe. This means
that the verbs have no side effects. The advantage that the user
never worries about creating copies to avoid contaminating the
input dataframe. It is normal in most data-analysis workflows for
the user to manipulate the data many times in different ways. It is
also the case that most datasets are small enough that they can be
copied many times with no noticeable effect on performance.

If you have a dataset small enough to fit in memory but too large
to copy all the time without affecting performance, then consider
using the :obj:`~plydata.options.modify_input_data` option. However,
only a few verbs can modify the input data and it is noted in
the documentation.

Datastore Support
-----------------

Single Table Verbs

+-------------------------------------------------+-----------+--------+
| Verb                                            | Dataframe | Sqlite |
+=================================================+===========+========+
| :class:`~plydata.one_table_verbs.arrange`       | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.create`        | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.define`        | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.distinct`      | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.do`            | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.group_by`      | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.group_indices` | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.head`          | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.pull`          | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.query`         | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.rename`        | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.sample_frac`   | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.sample_n`      | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.select`        | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.slice_rows`    | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.summarize`     | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.tail`          | Yes       | No     |
+-------------------------------------------------+-----------+--------+
| :class:`~plydata.one_table_verbs.ungroup`       | Yes       | No     |
+-------------------------------------------------+-----------+--------+

Helper verbs

+------------------------------------------------+-----------+--------+
| Verb                                           | Dataframe | Sqlite |
+================================================+===========+========+
| :class:`~plydata.helper_verbs.count`           | Yes       | No     |
+------------------------------------------------+-----------+--------+
| :class:`~plydata.helper_verbs.tally`           | Yes       | No     |
+------------------------------------------------+-----------+--------+
| :class:`~plydata.helper_verbs.add_count`       | Yes       | No     |
+------------------------------------------------+-----------+--------+
| :class:`~plydata.helper_verbs.add_tally`       | Yes       | No     |
+------------------------------------------------+-----------+--------+
| :class:`~plydata.helper_verbs.call`            | Yes       | No     |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.arrange_all`,   |           |        |
|  :class:`~plydata.helper_verbs.arrange_at`,    | Yes       | No     |
|  :class:`~plydata.helper_verbs.arrange_if`     |           |        |
+------------------------------------------------+-----------+--------+
| :class:`~plydata.helper_verbs.create_all`,     |           |        |
| :class:`~plydata.helper_verbs.create_at`,      | Yes       | No     |
| :class:`~plydata.helper_verbs.create_if`       |           |        |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.group_by_all`,  |           |        |
|  :class:`~plydata.helper_verbs.group_by_at`,   | Yes       | No     |
|  :class:`~plydata.helper_verbs.group_by_if`    |           |        |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.mutate_all`,    |           |        |
|  :class:`~plydata.helper_verbs.mutate_at`,     | Yes       | No     |
|  :class:`~plydata.helper_verbs.mutate_if`      |           |        |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.query_all`,     |           |        |
|  :class:`~plydata.helper_verbs.query_at`,      | Yes       | No     |
|  :class:`~plydata.helper_verbs.query_if`       |           |        |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.rename_all`,    |           |        |
|  :class:`~plydata.helper_verbs.rename_at`,     | Yes       | No     |
|  :class:`~plydata.helper_verbs.rename_if`      |           |        |
+------------------------------------------------+-----------+--------+
|  :class:`~plydata.helper_verbs.summarize_all`, |           |        |
|  :class:`~plydata.helper_verbs.summarize_at`,  |  Yes      | No     |
|  :class:`~plydata.helper_verbs.summarize_if`   |           |        |
+------------------------------------------------+-----------+--------+

Two table verbs

+----------------------------------------------+-----------+--------+
| Verb                                         | Dataframe | Sqlite |
+==============================================+===========+========+
| :class:`~plydata.two_table_verbs.anti_join`  | Yes       | No     |
+----------------------------------------------+-----------+--------+
| :class:`~plydata.two_table_verbs.inner_join` | Yes       | No     |
+----------------------------------------------+-----------+--------+
| :class:`~plydata.two_table_verbs.left_join`  | Yes       | No     |
+----------------------------------------------+-----------+--------+
| :class:`~plydata.two_table_verbs.outer_join` | Yes       | No     |
+----------------------------------------------+-----------+--------+
| :class:`~plydata.two_table_verbs.right_join` | Yes       | No     |
+----------------------------------------------+-----------+--------+
| :class:`~plydata.two_table_verbs.semi_join`  | Yes       | No     |
+----------------------------------------------+-----------+--------+

.. _zen of python: https://www.python.org/dev/peps/pep-0020/
