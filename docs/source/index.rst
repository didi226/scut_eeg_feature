.. SCUT EEG FEATURE documentation master file, created by
   sphinx-quickstart on Wed Jul 17 11:41:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
:html_theme.sidebar_secondary.remove:

.. title:: SEFEA

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">h1 {display:none;}</style>

SEFEA-Python Homepage
======================

.. LOGO

.. image:: _static/logo.svg
   :alt: SEFEA-Python
   :class: logo, mainlogo, only-light
   :align: center

.. image:: _static/logo-dark.svg
   :alt: SEFEA-Python
   :class: logo, mainlogo, only-dark
   :align: center

.. rst-class:: h4 text-center font-weight-light my-4

   Open-source Python package for claculating different kinds of features for EEG or other time series signals.

.. frontpage gallery is added by a conditional in _templates/layout.html

=======================================================

.. toctree::
   :glob:
   :hidden:
   :includehidden:
   :maxdepth: 1


   Install<install>
   Example<example/index>
   API<api/index>
   Contribute<contribute>

