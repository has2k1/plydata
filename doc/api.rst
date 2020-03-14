.. _api:

#############
API Reference
#############

One table verbs
===============

.. currentmodule:: plydata.one_table_verbs

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   arrange
   create
   define
   distinct
   do
   group_by
   group_indices
   head
   mutate
   pull
   query
   rename
   sample_frac
   sample_n
   select
   slice_rows
   summarize
   tail
   transmute
   ungroup
   unique

Helpers
-------

.. currentmodule:: plydata.helper_verbs

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   add_count
   add_tally
   count
   tally
   call
   arrange_all
   arrange_at
   arrange_if
   create_all
   create_at
   create_if
   group_by_all
   group_by_at
   group_by_if
   mutate_all
   mutate_at
   mutate_if
   query_all
   query_at
   query_if
   rename_all
   rename_at
   rename_if
   select_all
   select_at
   select_if
   summarise_all
   summarise_at
   summarise_if
   summarize_all
   summarize_at
   summarize_if
   transmute_all
   transmute_at
   transmute_if


Two table verbs
===============

.. currentmodule:: plydata.two_table_verbs

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   anti_join
   full_join
   inner_join
   left_join
   outer_join
   right_join
   semi_join


Expression helpers
==================
These classes can be used construct complicated conditional assignment
expressions.

.. currentmodule:: plydata.expressions

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   case_when
   if_else

Options
=======

.. currentmodule:: plydata.options

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   modify_input_data
   get_option
   set_option
   options

Useful Functions
================

.. currentmodule:: plydata.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   Q
   n
   first2
   last2
   ply

Tidy Verbs
==========
These verbs help create `tidy data <https://en.wikipedia.org/wiki/Tidy_data>`_.
You can import them with :py:`from plydata.tidy import *`.

.. currentmodule:: plydata.tidy

Pivoting
--------
Pivoting changes the representation of a rectangular dataset, without changing
the data inside it.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   gather
   pivot_longer
   pivot_wider
   spread

String Columns
--------------
These verbs help separate multiple variables that are joined together in a
single column.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   extract
   separate
   separate_rows
   unite

Categorical Tools
=================

Functions to solve common problems when working with categorical variables.
You can import them with :py:`from plydata.cat_tools import *`.

Change the order of categories
------------------------------
These functions keep the values the same but change the order of the categories.

.. currentmodule:: plydata.cat_tools

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   cat_infreq
   cat_inorder
   cat_inseq
   cat_relevel
   cat_reorder
   cat_reorder2
   cat_rev
   cat_shift
   cat_shuffle

Change the value of categories
------------------------------
These functions change the categories while preserving the order (as much as possible).


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   cat_anon
   cat_collapse
   cat_lump
   cat_lump_lowfreq
   cat_lump_min
   cat_lump_n
   cat_lump_prop
   cat_other
   cat_recode
   cat_relabel
   cat_rename

Add or Remove Categories
------------------------
These functions leave the data values as is, but they add or remove categories.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   cat_drop
   cat_expand
   cat_explicit_na
   cat_remove_unused
   cat_unify

Combine Multiple Categoricals
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   cat_concat
   cat_zip


Datasets
========

These datasets ship with plydata and you can import them with from the
``plydata.data`` sub-package.

.. currentmodule:: plydata.data

.. autosummary::
   :toctree: generated/
   :template: data.rst

   fish_encounters
   gss_cat
