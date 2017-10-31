.. _api:

#############
API Reference
#############

Single table verbs
==================

.. currentmodule:: plydata.verbs

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   arrange
   call
   create
   define
   define_where
   distinct
   do
   dropna
   fillna
   group_by
   group_indices
   head
   modify_where
   mutate
   query
   rename
   sample_frac
   sample_n
   select
   summarize
   tail
   transmute
   ungroup
   unique

Helpers
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   add_count
   add_tally
   count
   tally
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

n()
---
This is a special internal function that returns the number of
rows in the current groups. It can be used in verbs like
:class:`~plydata.verbs.summarize`, :class:`~plydata.verbs.define`.
and :class:`~plydata.verbs.create`.
