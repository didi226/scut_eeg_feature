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

   
   UM3EST Third-order cumulants.
   Should be invoked via "CUMEST" for proper parameter checks
   :param y: input data vector (column)

       maxlag: maximum lag to be computed
     samp_seg: samples per segment
      overlap: percentage overlap of segments
        flag : 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.
           k1: the fixed lag in c3(m,k1): see below

   Output:
        y_cum:  estimated third-order cumulant,
                C3(m,k1)  -maxlag <= m <= maxlag















   ..
       !! processed by numpydoc !!

