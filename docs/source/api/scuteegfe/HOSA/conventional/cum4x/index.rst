scuteegfe.HOSA.conventional.cum4x
=================================

.. py:module:: scuteegfe.HOSA.conventional.cum4x


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.conventional.cum4x.cum4x
   scuteegfe.HOSA.conventional.cum4x.test


Module Contents
---------------

.. py:function:: cum4x(w, x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0, k2=0)

   
   Fourth-order cross-cumulants.
   :param w:          if w,x,y,z are matrices, rather than vectors, columns are
                      assumed to correspond to independent realizations,
                      overlap is set to 0, and samp_seg to the row dimension.
             maxlag - maximum lag to be computed    [default = 0]
   :param x:          if w,x,y,z are matrices, rather than vectors, columns are
                      assumed to correspond to independent realizations,
                      overlap is set to 0, and samp_seg to the row dimension.
             maxlag - maximum lag to be computed    [default = 0]
   :param y:          if w,x,y,z are matrices, rather than vectors, columns are
                      assumed to correspond to independent realizations,
                      overlap is set to 0, and samp_seg to the row dimension.
             maxlag - maximum lag to be computed    [default = 0]
   :param z  - data vectors/matrices with identical dimensions:          if w,x,y,z are matrices, rather than vectors, columns are
                                                                         assumed to correspond to independent realizations,
                                                                         overlap is set to 0, and samp_seg to the row dimension.
                                                                maxlag - maximum lag to be computed    [default = 0]

     samp_seg - samples per segment  [default = data_length]
      overlap - percentage overlap of segments [default = 0]
                overlap is clipped to the allowed range of [0,99].
        flag : 'biased', biased estimates are computed  [default]
               'unbiased', unbiased estimates are computed.
       k1,k2 : the fixed lags in C4(m,k1,k2) defaults to 0

   Output:
        y_cum:  estimated fourth-order cross cumulant,
              c4(t1,t2,t3) := cum( w^*(t), x(t+t1), y(t+t2), z^*(t+t3) )















   ..
       !! processed by numpydoc !!

.. py:function:: test()

