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

   
   CUM2EST Covariance function.
   Should be involed via "CUMEST" for proper parameter checks.
   :param y: input data vector (column)

       maxlag: maximum lag to be computed
     samp_seg: samples per segment (<=0 means no segmentation)
      overlap: percentage overlap of segments
         flag: 'biased', biased estimates are computed
               'unbiased', unbiased estimates are computed.

   Output:
        y_cum: estimated covariance,
               C2(m)  -maxlag <= m <= maxlag















   ..
       !! processed by numpydoc !!

