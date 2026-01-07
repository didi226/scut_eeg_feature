scuteegfe.HOSA.conventional.bispectrumi
=======================================

.. py:module:: scuteegfe.HOSA.conventional.bispectrumi


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.bispectrumi.bispectrumi
   scuteegfe.HOSA.conventional.bispectrumi.test


Module Contents
---------------

.. py:function:: bispectrumi(y, nlag=None, nsamp=None, overlap=None, flag='biased', nfft=None, wind=None)

   
   Indirect (lag-domain) method for estimating the bispectrum.

   :param y: data vector or time-series
   :param nlag: number of lags to compute [must be specified]
   :param nsamp: samples per segment [default: row dimension of y]
   :param overlap: percentage overlap [default = 0]
   :param flag: 'biased' or 'unbiased' [default is 'unbiased']
   :param nfft: FFT length to use [default = 128]
   :param wind: window function to apply
                if wind = 0, the Parzen window is applied (default)
                otherwise the hexagonal window with unity values is applied

   :returns:

             estimated bispectrum, an nfft x nfft array, with origin
                 at the center, and axes pointing down and to the right
             waxis: frequency-domain axis associated with the bispectrum
                 the i-th row (or column) of Bspec corresponds to f1 (or f2)
                 value of waxis(i)
   :rtype: Bspec















   ..
       !! processed by numpydoc !!

.. py:function:: test()

