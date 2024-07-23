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

   
   CUM4EST Fourth-order cumulants.
   :param Should be invoked via CUMEST for proper parameter checks:
   :param y_cum = cum4est:
   :type y_cum = cum4est: y, maxlag, samp_seg, overlap, flag, k1, k2
   :param Computes sample estimates of fourth-order cumulants:
   :param via the overlapped segment method.:
   :param y_cum = cum4est:      y: input data vector (column)
                           maxlag: maximum lag
   :type y_cum = cum4est: y, maxlag, samp_seg, overlap, flag, k1, k2
   :param samp_seg: samples per segment
                    overlap: percentage overlap of segments
                      flag : 'biased', biased estimates are computed
                           : 'unbiased', unbiased estimates are computed.

       k1,k2 : the fixed lags in C3(m,k1) or C4(m,k1,k2)

   Output:
       y_cum : estimated fourth-order cumulant slice
               C4(m,k1,k2)  -maxlag <= m <= maxlag















   ..
       !! processed by numpydoc !!

