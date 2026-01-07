scuteegfe.HOSA.conventional.cum2est
===================================

.. py:module:: scuteegfe.HOSA.conventional.cum2est


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum2est.cum2est


Module Contents
---------------

.. py:function:: cum2est(y, maxlag, nsamp, overlap, flag)

   
   Estimate the second-order cumulant (covariance) function.

   This function implements CUM2EST and should be invoked via ``cumest``
   for proper parameter checking.

   :param y: input data vector (column)
   :param maxlag: maximum lag to be computed
   :param samp_seg: samples per segment (<= 0 means no segmentation)
   :param overlap: percentage overlap of segments
   :param flag: covariance estimation flag
                'biased': biased estimates are computed
                'unbiased': unbiased estimates are computed

   :returns:

             estimated covariance sequence
                 C2(m), where -maxlag <= m <= maxlag
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

