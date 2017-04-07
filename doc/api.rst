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
   distinct
   do
   group_by
   group_indices
   head
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


Two table verbs
===============

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: main.rst

   full_join
   inner_join
   left_join
   outer_join
   right_join

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
