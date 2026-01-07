scuteegfe.HOSA.conventional.bispectrumd
=======================================

.. py:module:: scuteegfe.HOSA.conventional.bispectrumd


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bispectrumd.bispectrumd


Module Contents
---------------

.. py:function:: bispectrumd(y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating the bispectrum.

   :param y: Data vector or time-series.
   :type y: array-like
   :param nfft: FFT length. Default is the next power of two greater than the
                number of samples.
   :type nfft: int, optional
   :param wind: Window specification for frequency-domain smoothing.

                If ``wind`` is a scalar, it specifies the length of the side
                of the square for the Rao-Gabr optimal window (default is 5).

                If ``wind`` is a vector, a 2D window is calculated as::

                    w2(i, j) = wind(i) * wind(j) * wind(i + j)

                If ``wind`` is a matrix, it specifies the 2-D filter directly.
   :type wind: int or array-like, optional
   :param nsamp: Number of samples per segment. Default is chosen such that
                 there are 8 segments.

                 If ``y`` is a matrix, ``nsamp`` is set to the number of rows.
   :type nsamp: int, optional
   :param overlap: Percentage overlap between segments. Default is 50.

                   If ``y`` is a matrix, overlap is set to 0.
   :type overlap: float, optional

   :returns: * **Bspec** (*ndarray*) -- Estimated bispectrum, an ``nfft × nfft`` array, with the origin
               at the center and axes pointing down and to the right.
             * **waxis** (*ndarray*) -- Vector of frequencies associated with the rows and columns of
               ``Bspec``. The sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

