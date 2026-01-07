#!/usr/bin/env python


import os

import numpy as np


def nextpow2(num):
    """
    Return the next highest power of two greater than a given value.

    Args:
       num: Input value.

    Returns:
       The smallest power of two that is strictly greater than ``num``.

    Examples:
       >>> nextpow2(1000)
       1024
       >>> nextpow2(1024)
       2048

    Notes:
       Taken from:
       https://github.com/alaiacano/frfft/blob/master/frfft.py
    """

    npow = 2
    while npow <= num:
        npow = npow * 2
    return npow


def flat_eq(x, y):
    """
    Emulate MATLAB-style linear indexing assignment.

    This function emulates the MATLAB operation ``x(:) = y``.

    Args:
        x: Input array whose shape is preserved.
        y: Array whose elements are assigned to ``x`` in column-major order.

    Returns:
        An array with the same shape as ``x`` after linear assignment.
    """
    z = x.reshape(1, -1)
    z = y
    return z.reshape(x.shape)


def make_arr(arrs, axis=0):
    """
    Create arrays in a MATLAB-like concatenation manner.

    This function mimics MATLAB-style array construction by concatenating
    scalars, vectors, or matrices along a specified axis.

    Args:
      arrs: Iterable of scalars, vectors, or arrays to be concatenated.
      axis: Axis along which the arrays are concatenated.

    Returns:
      A NumPy array formed by concatenating the inputs in ``arrs``.

    Examples:
      Python equivalent of MATLAB-style array creation:
      >>> make_arr((4, range(1, 10)), axis=0)
      array([[4],
             [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    """
    a = []
    ctr = 0
    for x in arrs:
        if len(np.shape(x)) == 0:
            a.append(np.array([[x]]))
        elif len(np.shape(x)) == 1:
            a.append(np.array([x]))
        else:
            a.append(x)
        ctr += 1
    return np.concatenate(a, axis)


def shape(o, n):
    """
    Return a shape tuple padded to a given dimensionality.

    This function mimics MATLAB-style shape behavior by padding the
    shape of an array with ones if its dimensionality is less than ``n``.

    Args:
        o: Input array-like object.
        n: Desired number of dimensions.

    Returns:
        A tuple representing the shape of ``o`` with length ``n``.
    """
    s = o.shape
    if len(s) < n:
        x = tuple(np.ones(n - len(s)))
        return s + x
    else:
        return s


def here(f=__file__):
    """
    Return the directory containing a given file.

    By default, this returns the directory of the current script.

    Args:
        f: File path. Defaults to the current file.

    Returns:
        Absolute path of the directory containing ``f``.
    """
    return os.path.dirname(os.path.realpath(f))
