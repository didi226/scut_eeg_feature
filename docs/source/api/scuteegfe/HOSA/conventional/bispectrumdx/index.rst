scuteegfe.HOSA.conventional.bispectrumdx
========================================

.. py:module:: scuteegfe.HOSA.conventional.bispectrumdx


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bispectrumdx.bispectrumdx


Module Contents
---------------

.. py:function:: bispectrumdx(x, y, z, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating the cross-bispectrum.

   :param x: Data vector or time-series.
   :type x: array-like
   :param y: Data vector or time-series with the same dimensions as ``x``.
   :type y: array-like
   :param z: Data vector or time-series with the same dimensions as ``x``.
   :type z: array-like
   :param nfft: FFT length. Default is the next power of two greater than
                the segment length.
   :type nfft: int, optional
   :param wind: Window specification for frequency-domain smoothing.

                If ``wind`` is a scalar, it specifies the length of the side
                of the square for the Rao-Gabr optimal window (default is 5).

                If ``wind`` is a vector, a 2D window is calculated as::

                    w2(i, j) = wind(i) * wind(j) * wind(i + j)

                If ``wind`` is a matrix, it specifies the 2-D filter directly.
   :type wind: int or array-like, optional
   :param segsamp: Number of samples per segment. Default is chosen such that
                   there are 8 segments.

                   If ``x`` is a matrix, ``segsamp`` is set to the number of rows.
   :type segsamp: int, optional
   :param overlap: Percentage overlap between segments, in the range [0, 99].
                   Default is 50.

                   If ``x`` is a matrix, overlap is set to 0.
   :type overlap: float, optional

   :returns: * **Bspec** (*ndarray*) -- Estimated cross-bispectrum, an ``nfft × nfft`` array, with the
               origin at the center and axes pointing down and to the right.
             * **waxis** (*ndarray*) -- Vector of frequencies associated with the rows and columns of
               ``Bspec``. The sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

