scuteegfe.features.pdc_dtf
==========================

.. py:module:: scuteegfe.features.pdc_dtf

.. autoapi-nested-parse::

   Implements Partial Directed Coherence and Direct Transfer Function
   using MVAR processes.

   Reference
   ---------
   Luiz A. Baccala and Koichi Sameshima. Partial directed coherence:
   a new concept in neural structure determination.
   Biological Cybernetics, 84(6):463:474, 2001.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   scuteegfe.features.pdc_dtf.A


Functions
---------

.. autoapisummary::

   scuteegfe.features.pdc_dtf.calculate_dtf_pdc
   scuteegfe.features.pdc_dtf.compute_order
   scuteegfe.features.pdc_dtf.cov
   scuteegfe.features.pdc_dtf.dtf
   scuteegfe.features.pdc_dtf.mvar_fit
   scuteegfe.features.pdc_dtf.mvar_generate
   scuteegfe.features.pdc_dtf.pdc
   scuteegfe.features.pdc_dtf.plot_all
   scuteegfe.features.pdc_dtf.spectral_density


Module Contents
---------------

.. py:function:: calculate_dtf_pdc(X, sfreq=250, kind='dtf', p=None, normalize_=True, filter_bank=None)

   
   Calculate the Direct Transfer Function (DTF) or Partial Directed Coherence (PDC)
   from multivariate time series data.

   :param X: Time series data with shape (n_channel, n_times).
   :type X: ndarray
   :param sfreq: Sampling frequency in Hz. Default is 250.
   :type sfreq: int
   :param kind: Type of measure to compute. Options are 'dtf' for Direct Transfer Function
                or 'pdc' for Partial Directed Coherence. Default is 'dtf'.
   :type kind: str
   :param p: Order of the AR model. If None, the order will be estimated using BIC.
   :type p: int, optional
   :param normalize_: Whether to normalize the resulting matrix. Default is True.
   :type normalize_: bool
   :param filter_bank: Frequency range to filter. If provided, it should be a tuple
                       (low_freq, high_freq). The output will be filtered to
                       include only the frequencies within this range.
   :type filter_bank: tuple, optional

   :returns: The computed matrix (DTF or PDC) with shape (n_channel, n_channel).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_order(X, p_max)

   
   Estimate the AR order using Bayesian Information Criterion (BIC).

   :param X: Time series data of shape (N, n).
   :type X: ndarray
   :param p_max: Maximum model order to test.
   :type p_max: int

   :returns: Estimated order (int) and BIC values for orders from 0 to p_max (ndarray of shape (p_max + 1,)).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: cov(X, p)

   
   Compute vector autocovariance up to order p.

   :param X: Time series data of shape (N, n).
   :type X: ndarray
   :param p: Order of the model.
   :type p: int

   :returns: Autocovariance up to order p with shape (p + 1, N, N).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: dtf(A, sigma=None, n_fft=None, sfreq=250)

   
   Compute the Direct Transfer Function (DTF).

   :param A: AR coefficients with shape (p, N, N).
   :type A: ndarray
   :param sigma: Noise for each time series. Defaults to None.
   :type sigma: array, optional
   :param n_fft: Length of the FFT. Defaults to None.
   :type n_fft: int, optional
   :param sfreq: Sampling frequency. Default is 250.
   :type sfreq: int

   :returns: DTF matrix (ndarray of shape (n_fft, N, N)) and frequencies (ndarray of shape (n_fft,)).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: mvar_fit(X, p)

   
   Fit an MVAR model of order p using the Yule-Walker method.

   :param X: Time series data of shape (N, n).
   :type X: ndarray
   :param p: Order of the model.
   :type p: int

   :returns: AR coefficients (ndarray of shape (p, N, N)) and noise (array of shape (N,)).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: mvar_generate(A, n, sigma, burnin=500)

   
   Simulate a Multivariate AutoRegressive (MVAR) process.

   :param A: AR coefficients with shape (p, N, N), where N is the number of signals
             and p is the order of the model.
   :type A: ndarray
   :param n: Number of time samples.
   :type n: int
   :param sigma: Noise for each time series, shape (N,).
   :type sigma: array
   :param burnin: Length of the burn-in period (in samples). Default is 500.
   :type burnin: int

   :returns: Time series data of shape (N, n) after burn-in.
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: pdc(A, sigma=None, n_fft=None, sfreq=250)

   
   Compute Partial Directed Coherence (PDC).

   :param A: AR coefficients with shape (p, N, N).
   :type A: ndarray
   :param sigma: Noise for each time series. Defaults to None.
   :type sigma: array, optional
   :param n_fft: Length of the FFT. Defaults to None.
   :type n_fft: int, optional
   :param sfreq: Sampling frequency. Default is 250.
   :type sfreq: int

   :returns: PDC matrix (ndarray of shape (n_fft, N, N)) and frequencies (ndarray of shape (n_fft,)).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: plot_all(freqs, P, name)

   
   Plot a grid of subplots for visualizing the PDC or DTF matrices.

   :param freqs: Frequencies.
   :type freqs: ndarray
   :param P: PDC or DTF matrix of shape (n_fft, N, N).
   :type P: ndarray
   :param name: Title for the plot.
   :type name: str















   ..
       !! processed by numpydoc !!

.. py:function:: spectral_density(A, n_fft=None, sfreq=250)

   
   Estimate Power Spectral Density (PSD) from AR coefficients.

   :param A: AR coefficients with shape (p, N, N).
   :type A: ndarray
   :param n_fft: Length of the FFT. Defaults to None.
   :type n_fft: int, optional
   :param sfreq: Sampling frequency. Default is 250.
   :type sfreq: int

   :returns: Spectral density (ndarray of shape (n_fft, N, N)) and frequencies (ndarray of shape (n_fft,)).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:data:: A

