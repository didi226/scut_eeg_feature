scuteegfe.HOSA.conventional.bicoherencex
========================================

.. py:module:: scuteegfe.HOSA.conventional.bicoherencex


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bicoherencex.bicoherencex
   scuteegfe.HOSA.conventional.bicoherencex.test


Module Contents
---------------

.. py:function:: bicoherencex(w, x, y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating cross-bicoherence.

   :param w: data vector or time-series
   :param x: data vector or time-series (same dimensions as ``w``)
   :param y: data vector or time-series (same dimensions as ``w``)
   :param nfft: FFT length [default = power of two greater than nsamp]
                actual size used is the power of two greater than ``nsamp``
   :param wind: time-domain window applied to each data segment
                should be of length ``nsamp`` (see below); otherwise,
                the default Hanning window is used
   :param nsamp: samples per segment [default: such that there are 8 segments]
                 if the inputs are matrices, ``nsamp`` is set to the number of rows
   :param overlap: percentage overlap, allowed range [0, 99] [default = 50]
                   if the inputs are matrices, overlap is set to 0

   :returns:

             estimated cross-bicoherence, an nfft x nfft array, with
                 origin at the center, and axes pointing down and to the right
             waxis: vector of frequencies associated with the rows and columns
                 of ``bic``; sampling frequency is assumed to be 1
   :rtype: bic















   ..
       !! processed by numpydoc !!

.. py:function:: test()

