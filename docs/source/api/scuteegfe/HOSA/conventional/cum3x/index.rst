scuteegfe.HOSA.conventional.cum3x
=================================

.. py:module:: scuteegfe.HOSA.conventional.cum3x


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum3x.cum3x
   scuteegfe.HOSA.conventional.cum3x.test


Module Contents
---------------

.. py:function:: cum3x(x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0)

   
   Estimate the third-order cross-cumulant for a fixed lag.

   :param x: data vector or matrix
   :param y: data vector or matrix (same dimensions as x)
   :param z: data vector or matrix (same dimensions as x)
             if x, y, and z are matrices, columns correspond to independent
             realizations; overlap is set to 0 and nsamp is set to the
             row dimension
   :param maxlag: maximum lag to be computed [default = 0]
   :param nsamp: samples per segment [default = data length]
   :param overlap: percentage overlap of segments [default = 0]
                   overlap is clipped to the allowed range of [0, 99]
   :param flag: cumulant estimation flag
                'biased': biased estimates are computed [default]
                'unbiased': unbiased estimates are computed
   :param k1: fixed lag in the third-order cross-cumulant C3(m, k1) [default = 0]

   :returns:

             estimated third-order cross-cumulant sequence
                 E[x*(n) y(n + m) z(n + k1)], where -maxlag <= m <= maxlag
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

.. py:function:: test()

