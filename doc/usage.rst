Usage
=====

For data manipulation, there are two types of verbs;

1. :ref:`Single table verbs`
2. :ref:`Two table verbs`

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

+---------------------------------------+-----------+--------+
| Verb                                  | Dataframe | Sqlite |
+=======================================+===========+========+
| :class:`~plydata.verbs.arrange`       | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.count`         | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.create`        | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.define`        | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.define_where`  | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.distinct`      | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.do`            | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.group_by`      | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.group_indices` | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.head`          | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.modify_where`  | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.query`         | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.rename`        | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.sample_frac`   | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.sample_n`      | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.select`        | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.summarize`     | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.tail`          | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.tally`         | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.ungroup`       | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.anti_join`     | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.inner_join`    | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.left_join`     | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.outer_join`    | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.right_join`    | Yes       | No     |
+---------------------------------------+-----------+--------+
| :class:`~plydata.verbs.semi_join`     | Yes       | No     |
+---------------------------------------+-----------+--------+

.. _zen of python: https://www.python.org/dev/peps/pep-0020/
