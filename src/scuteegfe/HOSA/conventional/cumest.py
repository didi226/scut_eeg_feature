#!/usr/bin/env python

from __future__ import division

from .cum3est import *
from .cum4est import *


def cumest(y, norder=2, maxlag=0, nsamp=None, overlap=0, flag='biased', k1=0, k2=0):
    """
    Estimate second-, third-, or fourth-order cumulants.

    This function provides a unified interface for computing cumulants of
    order 2, 3, or 4 by dispatching to the corresponding estimation routines.

    Args:
        y: input time-series (vector)
        norder: cumulant order to compute (2, 3, or 4) [default = 2]
        maxlag: maximum cumulant lag to compute [default = 0]
        nsamp: samples per segment [default = data length]
        overlap: percentage overlap of segments [default = 0]
            overlap is clipped to the allowed range of [0, 99]
        flag: cumulant estimation flag
            'biased': biased estimates are computed [default]
            'unbiased': unbiased estimates are computed
        k1: fixed lag for third- or fourth-order cumulant slices
        k2: second fixed lag for fourth-order cumulant slices

    Returns:
        y_cum: estimated cumulant sequence or slice
            C2(m), C3(m, k1), or C4(m, k1, k2), where
            -maxlag <= m <= maxlag, depending on the selected order
    """

    (ksamp, nrecs) = y.shape
    if ksamp == 1:
        ksamp = nrecs
        nrecs = 1

    if norder < 2 or norder > 4:
        raise ValueError('cumulant order must be 2, 3 or 4')

    if maxlag < 0:
        raise ValueError('"maxlag" must be non-negative')

    if nrecs > 1: nsamp = ksamp
    if nsamp <= 0 or nsamp > ksamp: nsamp = ksamp

    if nrecs > 1: overlap = 0
    overlap = max(0, min(overlap, 99))

    # estimate the cumulants
    if norder == 2:
        y_cum = cum2est(y, maxlag, nsamp, overlap, flag)
    elif norder == 3:
        y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1)
    elif norder == 4:
        y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1, k2)

    return y_cum


def test():
    y = sio.loadmat(here(__file__) + '/demo/ma1.mat')['y']

    # The right results are:
    #           "biased": [-0.12250513  0.35963613  1.00586945  0.35963613 -0.12250513]
    #           "unbiaed": [-0.12444965  0.36246791  1.00586945  0.36246791 -0.12444965]
    print(cum2est(y, 2, 128, 0, 'unbiased'))
    print(cum2est(y, 2, 128, 0, 'biased'))

    # For the 3rd cumulant:
    #           "biased": [-0.18203039  0.07751503  0.67113035  0.729953    0.07751503]
    #           "unbiased": [-0.18639911  0.07874543  0.67641484  0.74153955  0.07937539]
    print(cum3est(y, 2, 128, 0, 'biased', 1))
    print(cum3est(y, 2, 128, 0, 'unbiased', 1))

    # For testing the 4th-order cumulant
    # "biased": [-0.03642083  0.4755026   0.6352588   1.38975232  0.83791117  0.41641134 -0.97386322]
    # "unbiased": [-0.04011388  0.48736793  0.64948927  1.40734633  0.8445089   0.42303979 -0.99724968]
    print(cum4est(y, 3, 128, 0, 'biased', 1, 1))
    print(cum4est(y, 3, 128, 0, 'unbiased', 1, 1))


if __name__ == '__main__':
    test()
