scuteegfe.HOSA.conventional.cum2x
=================================

.. py:module:: scuteegfe.HOSA.conventional.cum2x


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum2x.cum2x
   scuteegfe.HOSA.conventional.cum2x.test


Module Contents
---------------

.. py:function:: cum2x(x, y, maxlag=0, nsamp=0, overlap=0, flag='biased')

   
   Estimate the second-order cross-cumulant (cross-covariance) function.

   :param x: data vector or matrix
   :param y: data vector or matrix (same dimensions as x)
             if x and y are matrices, columns correspond to independent
             realizations; overlap is set to 0 and samp_seg is set to the
             row dimension
   :param maxlag: maximum lag to be computed [default = 0]
   :param nsamp: samples per segment [default = data length]
   :param overlap: percentage overlap of segments [default = 0]
                   overlap is clipped to the allowed range of [0, 99]
   :param flag: covariance estimation flag
                'biased': biased estimates are computed [default]
                'unbiased': unbiased estimates are computed

   :returns:

             estimated cross-covariance sequence
                 E[x*(n) y(n + m)], where -maxlag <= m <= maxlag
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

.. py:function:: test()

