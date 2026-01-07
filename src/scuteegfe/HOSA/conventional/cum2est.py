#!/usr/bin/env python

from __future__ import division

from ..tools.tools import nextpow2, flat_eq, here


def cum2est(y, maxlag, nsamp, overlap, flag):
    """
    Estimate the second-order cumulant (covariance) function.

    This function implements CUM2EST and should be invoked via ``cumest``
    for proper parameter checking.

    Args:
        y: input data vector (column)
        maxlag: maximum lag to be computed
        samp_seg: samples per segment (<= 0 means no segmentation)
        overlap: percentage overlap of segments
        flag: covariance estimation flag
            'biased': biased estimates are computed
            'unbiased': unbiased estimates are computed

    Returns:
        y_cum: estimated covariance sequence
            C2(m), where -maxlag <= m <= maxlag
    """

    (n1, n2) = shape(y, 2)
    N = n1 * n2
    overlap = np.fix(overlap / 100 * nsamp)
    nrecord = np.fix((N - overlap) / (nsamp - overlap))
    nadvance = nsamp - overlap

    y_cum = np.zeros([maxlag + 1, 1])
    ind = np.arange(nsamp)
    y = y.ravel(order='F')

    for i in np.arange(nrecord):
        x = y[ind]
        x = x - np.mean(x)

        for k in np.arange(maxlag + 1):
            y_cum[k] = y_cum[k] + np.dot(x[0:nsamp - k].T, x[k:nsamp])

        ind = ind + int(nadvance)

    if flag == 'biased':
        y_cum = y_cum / (nsamp * nrecord)
    else:
        y_cum = y_cum / (nrecord * (nsamp - np.matrix(range(maxlag + 1)).T))
        y_cum = np.asarray(y_cum)

    if maxlag > 0:
        y_cum = make_arr([np.conj(y_cum[maxlag:0:-1]), y_cum], axis=0)

    return y_cum
