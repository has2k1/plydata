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
   count
   create
   define
   define_where
   distinct
   do
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
   tally
   transmute
   ungroup
   unique


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
