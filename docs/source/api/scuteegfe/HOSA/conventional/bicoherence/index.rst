scuteegfe.HOSA.conventional.bicoherence
=======================================

.. py:module:: scuteegfe.HOSA.conventional.bicoherence


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bicoherence.calculate_bicoherence
   scuteegfe.HOSA.conventional.bicoherence.test


Module Contents
---------------

.. py:function:: calculate_bicoherence(y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating bicoherence
   :param y     - data vector or time-series:
   :param nfft - fft length [default = power of two > segsamp]: actual size used is power of two greater than 'nsamp'
   :param wind - specifies the time-domain window to be applied to each:      data segment; should be of length 'segsamp' (see below);
                                                                         otherwise, the default Hanning window is used.
   :param segsamp - samples per segment [default: such that we have 8 segments]
                                                  - if x is a matrix, segsamp is set to the number of rows
   :param overlap - percentage overlap:
                                        - if x is a matrix, overlap is set to 0.
   :param allowed range [0:
                            - if x is a matrix, overlap is set to 0.
   :param 99]. [default = 50];:
                                - if x is a matrix, overlap is set to 0.

   Output:
     bic     - estimated bicoherence: an nfft x nfft array, with origin
               at the center, and axes pointing down and to the right.
     waxis   - vector of frequencies associated with the rows and columns
               of bic;  sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

.. py:function:: test()

