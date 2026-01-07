import os
import numpy as np


def nextpow2(num):
    """
    Return the next highest power of two greater than a given value.

    Parameters
    ----------
    num : int
        Input value.

    Returns
    -------
    int
        The smallest power of two that is strictly greater than ``num``.

    Notes
    -----
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

    Parameters
    ----------
    x : ndarray
        Input array whose shape is preserved.
    y : ndarray
        Array whose elements are assigned to ``x`` in column-major order.

    Returns
    -------
    ndarray
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

    Parameters
    ----------
    arrs : iterable
        Iterable of scalars, vectors, or arrays to be concatenated.
    axis : int
        Axis along which the arrays are concatenated.

    Returns
    -------
    ndarray
        A NumPy array formed by concatenating the inputs in ``arrs``.

    Examples
    --------
    .. code-block:: python

        make_arr((4, range(1, 10)), axis=0)
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

    Parameters
    ----------
    o : array_like
        Input array-like object.
    n : int
        Desired number of dimensions.

    Returns
    -------
    tuple
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

    Parameters
    ----------
    f : str or pathlib.Path, optional
        File path. Defaults to the current file.

    Returns
    -------
    str
        Absolute path of the directory containing ``f``.
    """

    return os.path.dirname(os.path.realpath(f))
