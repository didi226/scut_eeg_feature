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

   
   Cross-covariance
   :param x: if x,y are matrices, rather than vectors, columns are
             assumed to correspond to independent realizations,
             overlap is set to 0, and samp_seg to the row dimension.
   :param y    - data vectors/matrices with identical dimensions: if x,y are matrices, rather than vectors, columns are
                                                                  assumed to correspond to independent realizations,
                                                                  overlap is set to 0, and samp_seg to the row dimension.
   :param maxlag - maximum lag to be computed    [default = 0]:

     samp_seg - samples per segment  [default = data_length]
      overlap - percentage overlap of segments [default = 0]
                overlap is clipped to the allowed range of [0,99].
        flag  - 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.

   Output:
        y_cum - estimated cross-covariance
                E x^*(n)y(n+m),   -maxlag <= m <= maxlag















   ..
       !! processed by numpydoc !!

.. py:function:: test()

