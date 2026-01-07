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
   :param y: Array whose elements are assigned to ``x`` in column-major order.

   :returns: An array with the same shape as ``x`` after linear assignment.















   ..
       !! processed by numpydoc !!

.. py:function:: here(f=__file__)

   
   Return the directory containing a given file.

   By default, this returns the directory of the current script.

   :param f: File path. Defaults to the current file.

   :returns: Absolute path of the directory containing ``f``.















   ..
       !! processed by numpydoc !!

.. py:function:: make_arr(arrs, axis=0)

   
   Create arrays in a MATLAB-like concatenation manner.

   This function mimics MATLAB-style array construction by concatenating
   scalars, vectors, or matrices along a specified axis.

   :param arrs: Iterable of scalars, vectors, or arrays to be concatenated.
   :param axis: Axis along which the arrays are concatenated.

   :returns: A NumPy array formed by concatenating the inputs in ``arrs``.

   .. rubric:: Examples

   Python equivalent of MATLAB-style array creation:
   >>> make_arr((4, range(1, 10)), axis=0)
   array([[4],
          [1, 2, 3, 4, 5, 6, 7, 8, 9]])















   ..
       !! processed by numpydoc !!

.. py:function:: nextpow2(num)

   
   Return the next highest power of two greater than a given value.

   :param num: Input value.

   :returns: The smallest power of two that is strictly greater than ``num``.

   .. rubric:: Examples

   >>> nextpow2(1000)
   1024
   >>> nextpow2(1024)
   2048

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
   :param n: Desired number of dimensions.

   :returns: A tuple representing the shape of ``o`` with length ``n``.















   ..
       !! processed by numpydoc !!

