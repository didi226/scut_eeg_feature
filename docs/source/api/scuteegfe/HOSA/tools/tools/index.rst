scuteegfe.HOSA.tools.tools
==========================

.. py:module:: scuteegfe.HOSA.tools.tools


Functions
---------

.. autoapisummary::

   scuteegfe.HOSA.tools.tools.flat_eq
   scuteegfe.HOSA.tools.tools.here
   scuteegfe.HOSA.tools.tools.make_arr
   scuteegfe.HOSA.tools.tools.nextpow2
   scuteegfe.HOSA.tools.tools.shape


Module Contents
---------------

.. py:function:: flat_eq(x, y)

   
   Emulate MATLAB-style linear indexing assignment.

   This function emulates the MATLAB operation ``x(:) = y``.

   :param x: Input array whose shape is preserved.
   :type x: ndarray
   :param y: Array whose elements are assigned to ``x`` in column-major order.
   :type y: ndarray

   :returns: An array with the same shape as ``x`` after linear assignment.
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: here(f=__file__)

   
   Return the directory containing a given file.

   By default, this returns the directory of the current script.

   :param f: File path. Defaults to the current file.
   :type f: str or pathlib.Path, optional

   :returns: Absolute path of the directory containing ``f``.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: make_arr(arrs, axis=0)

   
   Create arrays in a MATLAB-like concatenation manner.

   This function mimics MATLAB-style array construction by concatenating
   scalars, vectors, or matrices along a specified axis.

   :param arrs: Iterable of scalars, vectors, or arrays to be concatenated.
   :type arrs: iterable
   :param axis: Axis along which the arrays are concatenated.
   :type axis: int

   :returns: A NumPy array formed by concatenating the inputs in ``arrs``.
   :rtype: ndarray

   .. rubric:: Examples

   .. code-block:: python

       make_arr((4, range(1, 10)), axis=0)
       array([[4],
              [1, 2, 3, 4, 5, 6, 7, 8, 9]])















   ..
       !! processed by numpydoc !!

.. py:function:: nextpow2(num)

   
   Return the next highest power of two greater than a given value.

   :param num: Input value.
   :type num: int

   :returns: The smallest power of two that is strictly greater than ``num``.
   :rtype: int

   .. rubric:: Notes

   Taken from:
   https://github.com/alaiacano/frfft/blob/master/frfft.py















   ..
       !! processed by numpydoc !!

.. py:function:: shape(o, n)

   
   Return a shape tuple padded to a given dimensionality.

   This function mimics MATLAB-style shape behavior by padding the
   shape of an array with ones if its dimensionality is less than ``n``.

   :param o: Input array-like object.
   :type o: array_like
   :param n: Desired number of dimensions.
   :type n: int

   :returns: A tuple representing the shape of ``o`` with length ``n``.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

