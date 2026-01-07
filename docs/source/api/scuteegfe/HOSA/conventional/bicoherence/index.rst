scuteegfe.HOSA.conventional.bicoherence
=======================================

.. py:module:: scuteegfe.HOSA.conventional.bicoherence


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bicoherence.calculate_bicoherence


Module Contents
---------------

.. py:function:: calculate_bicoherence(y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating bicoherence.

   :param y: data vector or time-series
   :param nfft: FFT length [default = power of two greater than nsamp]
                actual size used is the power of two greater than ``nsamp``
   :param wind: time-domain window applied to each data segment
                should be of length ``segsamp`` (see below); otherwise,
                the default Hanning window is used
   :param nsamp: samples per segment [default: such that there are 8 segments]
                 if ``y`` is a matrix, ``nsamp`` is set to the number of rows
   :param overlap: percentage overlap, allowed range [0, 99] [default = 50]
                   if ``y`` is a matrix, overlap is set to 0

   :returns:

             estimated bicoherence, an nfft x nfft array, with origin
                 at the center, and axes pointing down and to the right
             waxis: vector of frequencies associated with the rows and columns
                 of ``bic``; sampling frequency is assumed to be 1
   :rtype: bic















   ..
       !! processed by numpydoc !!

