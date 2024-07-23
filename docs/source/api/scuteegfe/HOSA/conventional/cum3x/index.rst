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

   
   Third-order cross-cumulants.
   :param x: if x,y,z are matrices, rather than vectors, columns are
             assumed to correspond to independent realizations,
             overlap is set to 0, and samp_seg to the row dimension.
   :param y: if x,y,z are matrices, rather than vectors, columns are
             assumed to correspond to independent realizations,
             overlap is set to 0, and samp_seg to the row dimension.
   :param z  - data vectors/matrices with identical dimensions: if x,y,z are matrices, rather than vectors, columns are
                                                                assumed to correspond to independent realizations,
                                                                overlap is set to 0, and samp_seg to the row dimension.
   :param maxlag - maximum lag to be computed    [default = 0]:

     samp_seg - samples per segment  [default = data_length]
      overlap - percentage overlap of segments [default = 0]
                overlap is clipped to the allowed range of [0,99].
        flag : 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.
           k1: the fixed lag in c3(m,k1): defaults to 0

   Output:
        y_cum:  estimated third-order cross cumulant,
                E x^*(n)y(n+m)z(n+k1),   -maxlag <= m <= maxlag















   ..
       !! processed by numpydoc !!

.. py:function:: test()

