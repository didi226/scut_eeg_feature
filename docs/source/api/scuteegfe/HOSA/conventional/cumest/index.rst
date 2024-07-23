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

   
   Second-, third- or fourth-order cumulants.
   :param y - time-series  - should be a vector:

       norder - cumulant order: 2, 3 or 4 [default = 2]
       maxlag - maximum cumulant lag to compute [default = 0]
     samp_seg - samples per segment  [default = data_length]
      overlap - percentage overlap of segments [default = 0]
                overlap is clipped to the allowed range of [0,99].
        flag  - 'biased' or 'unbiased'  [default = 'biased']
       k1,k2  - specify the slice of 3rd or 4th order cumulants

   Output:
       y_cum  - C2(m) or C3(m,k1) or C4(m,k1,k2),  -maxlag <= m <= maxlag
                depending upon the cumulant order selected















   ..
       !! processed by numpydoc !!

.. py:function:: test()

