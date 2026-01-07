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

   
   Estimate the fourth-order cross-cumulant for fixed lags.

   :param w: data vector or matrix
   :param x: data vector or matrix (same dimensions as w)
   :param y: data vector or matrix (same dimensions as w)
   :param z: data vector or matrix (same dimensions as w)
             if w, x, y, and z are matrices, columns correspond to independent
             realizations; overlap is set to 0 and nsamp is set to the
             row dimension
   :param maxlag: maximum lag to be computed [default = 0]
   :param nsamp: samples per segment [default = data length]
   :param overlap: percentage overlap of segments [default = 0]
                   overlap is clipped to the allowed range of [0, 99]
   :param flag: cumulant estimation flag
                'biased': biased estimates are computed [default]
                'unbiased': unbiased estimates are computed
   :param k1: first fixed lag in the fourth-order cross-cumulant C4(m, k1, k2)
              [default = 0]
   :param k2: second fixed lag in the fourth-order cross-cumulant C4(m, k1, k2)
              [default = 0]

   :returns:

             estimated fourth-order cross-cumulant sequence
                 c4(t1, t2, t3) := cum(w*(t), x(t + t1), y(t + t2), z*(t + t3))
   :rtype: y_cum















   ..
       !! processed by numpydoc !!

.. py:function:: test()

