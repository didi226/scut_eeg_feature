scuteegfe.HOSA.conventional.bispectrumdx
========================================

.. py:module:: scuteegfe.HOSA.conventional.bispectrumdx


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bispectrumdx.bispectrumdx
   scuteegfe.HOSA.conventional.bispectrumdx.test


Module Contents
---------------

.. py:function:: bispectrumdx(x, y, z, nfft=None, wind=None, nsamp=None, overlap=None)

   
   :param x    - data vector or time-series:
   :param y    - data vector or time-series:
   :type y    - data vector or time-series: same dimensions as x
   :param z    - data vector or time-series:
   :type z    - data vector or time-series: same dimensions as x
   :param nfft - fft length [default = power of two > segsamp]:
   :param wind - window specification for frequency-domain smoothing:
                                                                      if 'wind' is a scalar, it specifies the length of the side
                                                                         of the square for the Rao-Gabr optimal window  [default=5]
                                                                      if 'wind' is a vector, a 2D window will be calculated via
                                                                         w2(i,j) = wind(i) * wind(j) * wind(i+j)
                                                                      if 'wind' is a matrix, it specifies the 2-D filter directly
   :param segsamp - samples per segment [default: such that we have 8 segments]
                                                  - if x is a matrix, segsamp is set to the number of rows
   :param overlap - percentage overlap:
                                        - if x is a matrix, overlap is set to 0.
   :param allowed range [0:
                            - if x is a matrix, overlap is set to 0.
   :param 99]. [default = 50];:
                                - if x is a matrix, overlap is set to 0.

   Output:
     Bspec   - estimated bispectrum: an nfft x nfft array, with origin
               at the center, and axes pointing down and to the right.
     waxis   - vector of frequencies associated with the rows and columns
               of Bspec;  sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

.. py:function:: test()

