scuteegfe.HOSA.conventional.cum3est
===================================

.. py:module:: scuteegfe.HOSA.conventional.cum3est


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum3est.cum3est


Module Contents
---------------

.. py:function:: cum3est(y, maxlag, nsamp, overlap, flag, k1)

   
   Estimate the third-order cumulant for a fixed lag.

   This function implements CUM3EST and should be invoked via ``cumest``
   for proper parameter checking.

   :param y: input data vector (column)
   :param maxlag: maximum lag to be computed
   :param nsamp: samples per segment
   :param overlap: percentage overlap of segments
   :param flag: cumulant estimation flag
                'biased': biased estimates are computed [default]
                'unbiased': unbiased estimates are computed
   :param k1: fixed lag in the third-order cumulant C3(m, k1)

   :returns:

             estimated third-order cumulant sequence
                 C3(m, k1), where -maxlag <= m <= maxlag
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

