scuteegfe.HOSA.conventional.bispectrumd
=======================================

.. py:module:: scuteegfe.HOSA.conventional.bispectrumd


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bispectrumd.bispectrumd
   scuteegfe.HOSA.conventional.bispectrumd.test


Module Contents
---------------

.. py:function:: bispectrumd(y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   :param y    - data vector or time-series:
   :param nfft - fft length [default = power of two > segsamp]:
   :param wind - window specification for frequency-domain smoothing:
                                                                      if 'wind' is a scalar, it specifies the length of the side
                                                                         of the square for the Rao-Gabr optimal window  [default=5]
                                                                      if 'wind' is a vector, a 2D window will be calculated via
                                                                         w2(i,j) = wind(i) * wind(j) * wind(i+j)
                                                                      if 'wind' is a matrix, it specifies the 2-D filter directly
   :param segsamp - samples per segment [default: such that we have 8 segments]
                                                  - if y is a matrix, segsamp is set to the number of rows
   :param overlap - percentage overlap [default = 50]:
                                                       - if y is a matrix, overlap is set to 0.

   Output:
     Bspec   - estimated bispectrum: an nfft x nfft array, with origin
               at the center, and axes pointing down and to the right.
     waxis   - vector of frequencies associated with the rows and columns
               of Bspec;  sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

.. py:function:: test()

