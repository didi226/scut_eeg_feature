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

   
   Emulate MATLAB's assignment of the form
   x(:) = y
















   ..
       !! processed by numpydoc !!

.. py:function:: here(f=__file__)

   
   This script's directory
















   ..
       !! processed by numpydoc !!

.. py:function:: make_arr(arrs, axis=0)

   
   Create arrays like MATLAB does
   python                                 MATLAB
   make_arr((4, range(1,10)), axis=0) => [4; 1:9]
















   ..
       !! processed by numpydoc !!

.. py:function:: nextpow2(num)

   
   Returns the next highest power of 2 from the given value.
   .. rubric:: Example

   >>nextpow2(1000)
   1024
   >>nextpow2(1024)
   2048

   Taken from: https://github.com/alaiacano/frfft/blob/master/frfft.py















   ..
       !! processed by numpydoc !!

.. py:function:: shape(o, n)

   
   Behave like MATLAB's shape
















   ..
       !! processed by numpydoc !!

