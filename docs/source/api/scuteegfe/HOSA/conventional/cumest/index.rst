scuteegfe.HOSA.conventional.cumest
==================================

.. py:module:: scuteegfe.HOSA.conventional.cumest


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cumest.cumest
   scuteegfe.HOSA.conventional.cumest.test


Module Contents
---------------

.. py:function:: cumest(y, norder=2, maxlag=0, nsamp=None, overlap=0, flag='biased', k1=0, k2=0)

   
   Estimate second-, third-, or fourth-order cumulants.

   This function provides a unified interface for computing cumulants of
   order 2, 3, or 4 by dispatching to the corresponding estimation routines.

   :param y: input time-series (vector)
   :param norder: cumulant order to compute (2, 3, or 4) [default = 2]
   :param maxlag: maximum cumulant lag to compute [default = 0]
   :param nsamp: samples per segment [default = data length]
   :param overlap: percentage overlap of segments [default = 0]
                   overlap is clipped to the allowed range of [0, 99]
   :param flag: cumulant estimation flag
                'biased': biased estimates are computed [default]
                'unbiased': unbiased estimates are computed
   :param k1: fixed lag for third- or fourth-order cumulant slices
   :param k2: second fixed lag for fourth-order cumulant slices

   :returns:

             estimated cumulant sequence or slice
                 C2(m), C3(m, k1), or C4(m, k1, k2), where
                 -maxlag <= m <= maxlag, depending on the selected order
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

.. py:function:: test()

