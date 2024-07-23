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

   
   Direct (FD) method for estimating cross-bicoherence
   :param w:
             - should have identical dimensions
   :param x:
             - should have identical dimensions
   :param y - data vector or time-series:
                                          - should have identical dimensions
   :param nfft - fft length [default = power of two > nsamp]: actual size used is power of two greater than 'nsamp'
   :param wind - specifies the time-domain window to be applied to each:      data segment; should be of length 'segsamp' (see below);
                                                                         otherwise, the default Hanning window is used.
   :param segsamp - samples per segment [default: such that we have 8 segments]
                                                  - if x is a matrix, segsamp is set to the number of rows
   :param overlap - percentage overlap:
                                        - if y is a matrix, overlap is set to 0.
   :param 0 to 99  [default = 50]:
                                   - if y is a matrix, overlap is set to 0.

   Output:
     bic     - estimated cross-bicoherence: an nfft x nfft array, with
               origin at center, and axes pointing down and to the right.
     waxis   - vector of frequencies associated with the rows and columns
               of bic;  sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

.. py:function:: test()

