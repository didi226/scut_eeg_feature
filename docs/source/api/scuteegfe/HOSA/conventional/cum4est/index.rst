scuteegfe.HOSA.conventional.cum4est
===================================

.. py:module:: scuteegfe.HOSA.conventional.cum4est


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum4est.cum4est


Module Contents
---------------

.. py:function:: cum4est(y, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0, k2=0)

   
   Estimate a fourth-order cumulant slice for fixed lags.

   This function implements CUM4EST and should be invoked via ``cumest``
   for proper parameter checking. It computes sample estimates of
   fourth-order cumulants using the overlapped segment method.

   :param y: input data vector (column)
   :param maxlag: maximum lag
   :param nsamp: samples per segment
   :param overlap: percentage overlap of segments
   :param flag: cumulant estimation flag
                'biased': biased estimates are computed
                'unbiased': unbiased estimates are computed
   :param k1: first fixed lag in the fourth-order cumulant C4(m, k1, k2)
   :param k2: second fixed lag in the fourth-order cumulant C4(m, k1, k2)

   :returns:

             estimated fourth-order cumulant slice
                 C4(m, k1, k2), where -maxlag <= m <= maxlag
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

