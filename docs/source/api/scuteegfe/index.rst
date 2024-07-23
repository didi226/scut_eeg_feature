scuteegfe
=========

.. py:module:: scuteegfe


Subpackages
-----------

.. toctree::
   :maxdepth: 1

   /api/scuteegfe/HOSA/index
   /api/scuteegfe/features/index
   /api/scuteegfe/mne_features_wrapper/index


Classes
-------

.. autoapisummary::

   scuteegfe.Feature


Functions
---------

.. autoapisummary::

   scuteegfe.Processing_inf_nan
   scuteegfe.Renyi_Entropy
   scuteegfe.Tsallis_Entropy
   scuteegfe.band_DE
   scuteegfe.band_Median_Frequency
   scuteegfe.bicoherence
   scuteegfe.calculate_dtf_pdc
   scuteegfe.compute_ARMA_kalman_filter
   scuteegfe.compute_Coherence
   scuteegfe.compute_DFA
   scuteegfe.compute_EMD
   scuteegfe.compute_Harmonic_Parameters
   scuteegfe.compute_Hilbert_abs
   scuteegfe.compute_Itakura_Distance
   scuteegfe.compute_Median_Frequency
   scuteegfe.compute_Num_zero_crossings
   scuteegfe.compute_Petrosian_fd
   scuteegfe.compute_Renyi_Entropy
   scuteegfe.compute_Shannon_entropy
   scuteegfe.compute_Tsallis_Entropy
   scuteegfe.compute_alpha_asymetry
   scuteegfe.compute_aperiodic_periodic_offset_exponent_cf
   scuteegfe.compute_correlation_dimension
   scuteegfe.compute_correlation_matrix
   scuteegfe.compute_cross_frequency_coupling
   scuteegfe.compute_detrended_fluctuation
   scuteegfe.compute_dispersion_entropy
   scuteegfe.compute_fuzzy_entropy
   scuteegfe.compute_hosa_bicoherence
   scuteegfe.compute_multiscale_permutation_entropy
   scuteegfe.compute_multiscale_sample_entropy
   scuteegfe.compute_offset_exponent_cf
   scuteegfe.compute_pac_connectivity
   scuteegfe.compute_periodic_pac_connectivity
   scuteegfe.compute_perm_entropy
   scuteegfe.compute_pow_freq_bands_cd
   scuteegfe.compute_pow_freq_bands_remove_aperiodic
   scuteegfe.compute_stft_2019
   scuteegfe.compute_test2
   scuteegfe.compute_wavelet_entropy
   scuteegfe.filter_bank
   scuteegfe.find_nearest
   scuteegfe.flatten_lower_triangle
   scuteegfe.get_fft_values
   scuteegfe.get_power_from_channel
   scuteegfe.imp_extract_fft
   scuteegfe.imp_extract_wavelet
   scuteegfe.pow_freq_bands_from_spectrum
   scuteegfe.reshape_to_lower_triangle


Package Contents
----------------

.. py:class:: Feature(data=None, sfreq=250, selected_funcs=None, funcs_params=None, n_jobs=1, ch_names=None, return_as_df=False, log_teager_kaiser_energy=False)

   .. py:method:: __getitem__(item)

      
      Slice the Feature instance on the sample dimension.

      :param item: Slice object for indexing the features array.
      :type item: slice

      :returns: A new instance of Feature with sliced features.
      :rtype: Feature















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__()

      
      Generate a string representation of the FeatureExtractor instance.

      If the features attribute is None, a prompt to input data is returned. Otherwise,
      the string representation includes the number of epochs, channels, and features,
      along with the feature names.

      :returns: String representation of the FeatureExtractor instance.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: copy()

      
      Copy the instance of Feature.

      :returns: A copy of the object.
      :rtype: Feature















      ..
          !! processed by numpydoc !!


   .. py:method:: feature_df2plot(features, feature_names, ch_names, sub_type='sub_type_1')
      :staticmethod:


      
      Convert features to a DataFrame for plotting.

      This method organizes features, feature names, and channel names into a pandas DataFrame,
      which can be used for visualization purposes, particularly with seaborn.

      :param features: The feature array with shape (n_samples, n_channels, n_features).
      :type features: ndarray
      :param feature_names: List of feature names.
      :type feature_names: list
      :param ch_names: List of channel names.
      :type ch_names: list
      :param sub_type: Description of the dataset type. Defaults to 'sub_type_1'.
      :type sub_type: str, optional

      :returns:

                DataFrame containing the features, channel names, type, and values,
                                  structured for plotting.
      :rtype: pandas.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: feature_smooth(data, smooth_type='lds', window_size=10)

      
      Smooth features using specified smoothing techniques.

      This function smooths the features using various smoothing techniques, including moving average filter,
      linear dynamical system (LDS) approach, and Unscented Kalman Filter. The method and window size for
      smoothing can be specified.

      :param data: The input data with shape (n_epoch, n_channel, n_feature).
      :type data: ndarray
      :param smooth_type: The type of smoothing technique to use. Options are:
                          - "mv_av_filter": Moving average filter
                          - "lds": Linear dynamic system (LDS) approach
                          - "UnscentedKalmanFilter": Unscented Kalman Filter
                          - "UnscentedKalmanFilter_sigmoid": Unscented Kalman Filter with sigmoid function
                          Defaults to "lds".
      :type smooth_type: str, optional
      :param window_size: The size of the processing window. Defaults to 10.
      :type window_size: int, optional

      :returns: The smoothed data.
      :rtype: ndarray

      .. rubric:: References

      Duan R N, Zhu J Y, Lu B L. Differential entropy feature for EEG-based emotion classification[C]//2013 6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013: 81-84.
      Shi L C, Lu B L. Off-line and on-line vigilance estimation based on linear dynamical system and manifold learning[C]//2010 Annual International Conference of the IEEE Engineering in Medicine and Biology. IEEE, 2010: 6587-6590.
      Zheng W L, Zhu J Y, Lu B L. Identifying stable patterns over time for emotion recognition from EEG[J]. IEEE Transactions on Affective Computing, 2017, 10(3): 417-429.















      ..
          !! processed by numpydoc !!


   .. py:method:: fix_missing()

      
      Fix missing values in the features.

      This method uses the mean strategy to impute missing values. If the mean strategy fails,
      it falls back to using a constant strategy with a fill value of 0.

      :returns: A new instance of FeatureExtractor with missing values imputed.
      :rtype: FeatureExtractor















      ..
          !! processed by numpydoc !!


   .. py:method:: fix_multi_feature(log=True)

      
      Fix multi-feature data by rearranging it.

      This function rearranges the multi-feature data to ensure proper alignment and, optionally,
      applies a logarithmic transformation to the 'teager_kaiser_energy' feature.

      :param log: If True, applies a logarithmic transformation to the 'teager_kaiser_energy' feature.
                  Defaults to True.
      :type log: bool, optional

      :returns: None















      ..
          !! processed by numpydoc !!


   .. py:method:: get_data(n_sample_list=None)

      
      Get the feature vector array.

      :param n_sample_list: List of sample indices to slice the features array. Defaults to None.
      :type n_sample_list: list, optional

      :returns: The original or sliced features array.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_funcs(selected_funcs)

      
      Retrieve custom feature decomposition functions based on selected function names.

      :param selected_funcs: List of feature function names or tuples of
                             (function name, function reference) to retrieve.
      :type selected_funcs: list of str or list of tuple

      :returns:

                A tuple containing:
                    - funcs (set of tuple): A set of tuples where each tuple is (function name, function reference).
                    - feature_names_order (list of str): List of feature function names in the order they were provided.
      :rtype: tuple

      :raises AttributeError: If the format of `selected_funcs` is not valid.















      ..
          !! processed by numpydoc !!


   .. py:method:: lsd_KalmanFilter(data, window_size)
      :staticmethod:


      
      Apply a Kalman Filter for smoothing data.

      This function uses a Kalman Filter to smooth the input data within a specified window size.

      :param data: The input data to be smoothed.
      :type data: array-like
      :param window_size: The size of the processing window.
      :type window_size: int

      :returns: The smoothed data.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: lsd_UnscentedKalmanFilter(data, window_size, observation_functions_type=None)
      :staticmethod:


      
      Apply an Unscented Kalman Filter for smoothing data.

      This function uses an Unscented Kalman Filter to smooth the input data within a specified window size.
      It supports optional sigmoid observation functions.

      :param data: The input data to be smoothed.
      :type data: array-like
      :param window_size: The size of the processing window.
      :type window_size: int
      :param observation_functions_type: The type of observation function. If 'sigmoid', applies a
                                         sigmoid-based observation function. Defaults to None.
      :type observation_functions_type: str, optional

      :returns: The smoothed data.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: moving_average_filter(data, window_size)
      :staticmethod:


      
      Apply a moving average filter to the data.

      This function applies a moving average filter with a specified window size to smooth the data.

      :param data: The input data to be filtered.
      :type data: array-like
      :param window_size: The size of the moving window.
      :type window_size: int

      :returns: The filtered data.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_feature_sns(Feature1, Feature2, ch_names, sub_type1='type1', sub_type2='type2')
      :staticmethod:


      
      Plot features using seaborn.

      :param Feature1: First Feature instance.
      :type Feature1: Feature
      :param Feature2: Second Feature instance.
      :type Feature2: Feature
      :param ch_names: List of channel names.
      :type ch_names: list
      :param sub_type1: Description of the first dataset type. Defaults to 'type1'.
      :type sub_type1: str, optional
      :param sub_type2: Description of the second dataset type. Defaults to 'type2'.
      :type sub_type2: str, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: reorder()

      
      Reorder features alphabetically by their names.

      :returns: A new instance of FeatureExtractor with features reordered alphabetically.
      :rtype: FeatureExtractor















      ..
          !! processed by numpydoc !!


   .. py:method:: ttest_feature(Feature1, Feature2, ch_names)
      :staticmethod:


      
      Run t-test and visualize p-values using a heatmap.

      This function performs a t-test between the features of two Feature instances and visualizes the p-values
      using a heatmap. The heatmap shows the -log10 of the p-values for better visualization.

      :param Feature1: An instance of the Feature class.
      :type Feature1: Feature
      :param Feature2: Another instance of the Feature class.
      :type Feature2: Feature
      :param ch_names: List of channel names.
      :type ch_names: list

      :returns: A tuple containing the t-statistics and p-values.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:attribute:: __feature_names
      :value: None



   .. py:attribute:: __features
      :value: None



   .. py:attribute:: __features


   .. py:attribute:: __features_fix
      :value: False



   .. py:attribute:: __features_raw


   .. py:attribute:: __list_multi_feature
      :value: ['teager_kaiser_energy0', 'spect_slope0', 'energy_freq_bands0', 'wavelet_coef_energy0',...



   .. py:attribute:: example_data


   .. py:property:: feature_names
      
      Get the feature names.

      :returns: Array of feature names.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: feature_names_order


   .. py:attribute:: features


   .. py:property:: features
      
      Get the features with possible multi-feature fixing.

      :returns: The features array.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: funcs


   .. py:attribute:: funcs_params


   .. py:attribute:: funcs_subset_no_spect_slope


   .. py:attribute:: log_teager_kaiser_energy


   .. py:attribute:: mne_defined_funcs


   .. py:attribute:: n_channel


.. py:function:: Processing_inf_nan(data)

   
   Processes infinite and NaN values in the data.

   :param data: Input data to process.
   :type data: ndarray

   :returns: Processed data with infinite and NaN values handled.
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: Renyi_Entropy(time_series, alpha)

   
   Compute the Renyi entropy of the sample data.

   :param time_series: Input time series data.
   :type time_series: Union[Vector, str]
   :param alpha: Entropy parameter.
   :type alpha: float

   :returns: Renyi entropy value.
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: Tsallis_Entropy(time_series, alpha=2)

   
   Compute the Tsallis entropy of the sample data.

   :param time_series: Input time series data.
   :type time_series: Union[Vector, str]
   :param alpha: Entropy parameter. Defaults to 2.
   :type alpha: float, optional

   :returns: Tsallis entropy value.
   :rtype: float

   .. rubric:: References

   Tsallis C. Possible generalization of Boltzmann-Gibbs statistics[J]. Journal of Statistical Physics, 1988, 52(1-2): 479-487.
   https://zhuanlan.zhihu.com/p/81462898 (Chinese reference explaining the base of the logarithm used)

   .. note:: There is a question about why the base of the logarithm used is 2.















   ..
       !! processed by numpydoc !!

.. py:function:: band_DE(Pxx, f, Par_ratios=1, band=None)

   
   Computes features from fixed frequency bands.

   :param Pxx: Power spectral density.
   :type Pxx: ndarray
   :param f: Frequency vector.
   :type f: ndarray
   :param Par_ratios: Whether to compute ratios (1) or not (0).
   :type Par_ratios: int
   :param band: Frequency bands to analyze, shape (fea_num, 2).
   :type band: ndarray

   :returns: Computed features.
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: band_Median_Frequency(Pxx, f, band=None)

   
   Compute the median frequency for a given power spectral density (PSD) and frequency band.

   :param Pxx: Power spectral density values.
   :type Pxx: array
   :param f: Frequency values.
   :type f: array
   :param band: Selected frequency bands. Defaults to None.
   :type band: ndarray, optional

   :returns: Median frequency values for each band.
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: bicoherence(y, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Direct (FD) method for estimating bicoherence
   :param y     - data vector or time-series:
   :param nfft - fft length [default = power of two > segsamp]: actual size used is power of two greater than 'nsamp'
   :param wind - specifies the time-domain window to be applied to each:      data segment; should be of length 'segsamp' (see below);
                                                                         otherwise, the default Hanning window is used.
   :param segsamp - samples per segment [default: such that we have 8 segments]
                                                  - if x is a matrix, segsamp is set to the number of rows
   :param overlap - percentage overlap:
                                        - if x is a matrix, overlap is set to 0.
   :param allowed range [0:
                            - if x is a matrix, overlap is set to 0.
   :param 99]. [default = 50];:
                                - if x is a matrix, overlap is set to 0.

   Output:
     bic     - estimated bicoherence: an nfft x nfft array, with origin
               at the center, and axes pointing down and to the right.
     waxis   - vector of frequencies associated with the rows and columns
               of bic;  sampling frequency is assumed to be 1.















   ..
       !! processed by numpydoc !!

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

.. py:function:: compute_ARMA_kalman_filter(data, AR_p=10, MA_q=1)

   
   Compute ARMA modeling coefficients using the Kalman filter.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param AR_p: Order of the AR model. Defaults to 10.
   :type AR_p: int, optional
   :param MA_q: Order of the MA model. Defaults to 1.
   :type MA_q: int, optional

   :returns: ARMA coefficients with shape (n_channel, AR_p + MA_q).
   :rtype: ndarray

   .. rubric:: Example

   >>> rng = np.random.RandomState(42)
   >>> n_epochs, n_channels, n_times = 2, 1, 1750
   >>> X = rng.randn(n_epochs, n_channels, n_times)
   >>> feat = Feature(X, sfreq=250, selected_funcs={'ARMA_kalman_filter'})
   >>> print(feat.features.shape)

   .. rubric:: References

   [1] Rossow A B, Salles E O T, Côco K F. Automatic sleep staging using a single-channel EEG modeling by Kalman filter
   and HMM[C]//ISSNIP Biosignals and Biorobotics Conference 2011. IEEE, 2011: 1-6.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Coherence(data, Co_channel=None, sfreq=250, band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]]))

   
   Compute coherence between different channels within specified frequency bands.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param Co_channel: Channels to compute coherence for. Shape (n_Co_channel, 2). Defaults to None.
   :type Co_channel: ndarray, optional
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param band: Frequency bands. Shape (fea_num, 2). Defaults to predefined bands.
   :type band: ndarray, optional

   :returns: Coherence features with shape (n_channel, n_channel * band_num).
   :rtype: ndarray

   .. rubric:: Notes

   For single-channel data, this function is not applicable.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_DFA(data, sfreq=250, win_times=1)

   
   Perform Detrended Fluctuation Analysis (DFA) to find long-term statistical correlations in a time series.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param win_times: Window duration in seconds. Defaults to 1.
   :type win_times: int, optional

   :returns: DFA features with shape (n_channels, section_num).
   :rtype: ndarray

   .. rubric:: References

   Peng C K, Havlin S, Stanley H E, Goldberger A L. Quantification of scaling exponents and crossover phenomena in
   nonstationary heartbeat time series[J]. Chaos: An Interdisciplinary Journal of Nonlinear Science, 1995, 5(1): 82-87.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_EMD(data, sfreq=250, EMD_times=1, EMD_params=6)

   
   Compute the Empirical Mode Decomposition (EMD) of the data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param EMD_times: Duration of each EMD segment in seconds. Defaults to 1.
   :type EMD_times: int, optional
   :param EMD_params: Number of Intrinsic Mode Functions (IMFs) to extract. Defaults to 6.
   :type EMD_params: int, optional

   :returns: EMD features with shape (n_channels, section_num * EMD_params * EMD_length).
   :rtype: ndarray

   .. rubric:: Notes

   - The EMD is applied to segments of the data, and the resulting IMFs are used as features.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Harmonic_Parameters(data, sfreq=250, band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]]))

   
   Compute harmonic parameters including center frequency, bandwidth, and spectral value at the center frequency.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param band: Frequency bands with shape (n_bands, 2). Defaults to predefined bands.
   :type band: ndarray, optional

   :returns: Harmonic parameters with shape (n_channels, n_band).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Hilbert_abs(data)

   
   Compute the absolute value of the Hilbert transform (envelope) of the data. (abandon)

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray

   :returns: Absolute value of the Hilbert transform with shape (n_channels,).
   :rtype: ndarray

   .. rubric:: Notes

   - This function is currently deprecated.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Itakura_Distance(data, baseline_data=None, dist='square', options={'max_slope': 2.0}, precomputed_cost=None, return_cost=False, return_accumulated=False, return_path=False)

   
   Compute the Itakura distance between the data and baseline data using dynamic time warping (DTW).

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param baseline_data: Baseline data with shape (n_channels, n_times). Defaults to None.
   :type baseline_data: ndarray, optional
   :param dist: Distance metric to use. Defaults to 'square'.
   :type dist: str or callable, optional
   :param options: Method options. Defaults to {'max_slope': 2.0}.
   :type options: dict, optional
   :param precomputed_cost: Precomputed cost matrix. Defaults to None.
   :type precomputed_cost: array-like, optional
   :param return_cost: If True, return the cost matrix. Defaults to False.
   :type return_cost: bool, optional
   :param return_accumulated: If True, return the accumulated cost matrix. Defaults to False.
   :type return_accumulated: bool, optional
   :param return_path: If True, return the optimal path. Defaults to False.
   :type return_path: bool, optional

   :returns: Itakura distance with shape (n_channels,).
   :rtype: ndarray

   .. rubric:: References

   https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html#pyts.metrics.dtw















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Median_Frequency(data, sfreq=250, band=np.array([[0.5, 2], [2, 4], [4, 5], [5, 7], [7, 10], [10, 13], [13, 15], [15, 20], [20, 30], [30, 40]]), N=None)

   
   Compute the median frequency for each channel and frequency band.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param band: Frequency bands. Defaults to predefined bands.
   :type band: ndarray, optional
   :param N: Number of samples to use. Defaults to None.
   :type N: int, optional

   :returns: Median frequency values with shape (n_channels, band_num).
   :rtype: ndarray

   .. rubric:: References

   [1] Gudmundsson S, Runarsson T P, Sigurdsson S. Automatic sleep staging using support vector machines with posterior probability estimates
       [C]//International Conference on Computational Intelligence for Modelling, Control and Automation and International Conference on Intelligent Agents,
       Web Technologies and Internet Commerce (CIMCA-IAWTIC'06). IEEE, 2005, 2: 366-372.
   [2] Thongpanja S, Phinyomark A, Phukpattaranont P, et al. Mean and median frequency of EMG signal to determine muscle force based on
       time-dependent power spectrum[J]. Elektronika ir Elektrotechnika, 2013, 19(3): 51-56.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Num_zero_crossings(data)

   
   Computes the number of zero crossings for each channel.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray

   :returns: Number of zero crossings per channel, shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Petrosian_fd(data)

   
   Computes the Petrosian fractal dimension for each channel.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray

   :returns: Fractal dimension per channel, shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Renyi_Entropy(data, sfreq=250, round_para=1, win_times=1, alpha=2)

   
   Compute the Renyi entropy for each channel using a sliding window approach.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param round_para: Number of decimal places to round the data. Defaults to 1.
   :type round_para: int, optional
   :param win_times: Window duration in seconds. Defaults to 1.
   :type win_times: int, optional
   :param alpha: Renyi entropy parameter. Defaults to 2.
   :type alpha: float, optional

   :returns: Computed Renyi entropy with shape (n_channels, section_num).
   :rtype: ndarray

   .. rubric:: Notes

   - The entropy is calculated for each window of data and then averaged across all windows.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Shannon_entropy(data, sfreq=250, round_para=1, win_times=1)

   
   Compute the Shannon entropy of the data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param round_para: Rounding precision for data. Defaults to 1.
   :type round_para: int, optional
   :param win_times: Window duration in seconds. Defaults to 1.
   :type win_times: int, optional

   :returns: Shannon entropy features with shape (n_channels, section_num).
   :rtype: ndarray

   .. rubric:: References

   Shannon C E. A mathematical theory of communication[J]. Bell System Technical Journal, 1948, 27(3): 379-423.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_Tsallis_Entropy(data, sfreq=250, round_para=1, win_times=1, alpha=2)

   
   Compute the Tsallis entropy for each channel using a sliding window approach.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param round_para: Number of decimal places to round the data. Defaults to 1.
   :type round_para: int, optional
   :param win_times: Window duration in seconds. Defaults to 1.
   :type win_times: int, optional
   :param alpha: Tsallis entropy parameter. Defaults to 2.
   :type alpha: float, optional

   :returns: Computed Tsallis entropy with shape (n_channels, section_num).
   :rtype: ndarray

   .. rubric:: Notes

   - The entropy is calculated for each window of data and then averaged across all windows.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_alpha_asymetry(data, sfreq=100, freq1=8, freq2=12, left='F3', right='F4', channel=None, mode='eeglab')

   
   Compute the alpha asymmetry between two specified channels.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the signal. Default is 100 Hz.
   :type sfreq: int
   :param freq1: Lower bound of the alpha frequency range.
   :type freq1: float
   :param freq2: Upper bound of the alpha frequency range.
   :type freq2: float
   :param left: Name of the left channel.
   :type left: str
   :param right: Name of the right channel.
   :type right: str
   :param channel: List of channel names. Default is None, in which case a default list is used.
   :type channel: list or None
   :param mode: Method for computing alpha asymmetry. Options are:
                - "eeglab": Method used in EEGLAB.
                - "definition_ln": Logarithmic difference.
                - "definition_ratio": Ratio difference.
                - "definition_lnratio": Logarithmic ratio difference.
                - "definition_ln_rel": Logarithmic relative difference.
                - "definition_ratio_rel": Ratio relative difference.
                - "definition_lnratio_rel": Logarithmic ratio relative difference.
   :type mode: str

   :returns: Array of alpha asymmetry values with shape (n_channels,).
   :rtype: ndarray

   .. rubric:: Notes

   - Computes alpha asymmetry using different methods depending on the `mode` parameter.
   - If `mode` is "eeglab", uses the power spectral density (PSD) of the specified channels.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_aperiodic_periodic_offset_exponent_cf(data, sfreq=250, n=1024, freq_range=None, method='welch')

   
   Compute aperiodic and periodic parameters of the power spectrum from EEG data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the signal. Default is 250 Hz.
   :type sfreq: int
   :param n: Number of frequency points for the Fourier transform. Typically set to the number of data points.
   :type n: int
   :param freq_range: Frequency range for fitting with FOOOF. Default is None.
   :type freq_range: list or None
   :param method: Method for computing the power spectrum. Options are:
                  - "fft": Fast Fourier Transform
                  - "welch": Welch's method
   :type method: str

   :returns: Flattened array of aperiodic and periodic parameters with shape (n_channels * 2,).
   :rtype: ndarray

   .. rubric:: Notes

   - Computes the offset and exponent of the aperiodic component and the periodic component of the power spectrum.
   - Uses the FOOOFGroup for fitting the power spectrum.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_correlation_dimension(data, emb_dim=10)

   
   :param data: ndarray,           shape (n_channels, n_times)
   :param emb_dim: int                嵌入维度默认为10

   Returns:         feature            shape (n_channels)















   ..
       !! processed by numpydoc !!

.. py:function:: compute_correlation_matrix(data, sfreq=250, kind='correlation', filter_bank=None, n_win=1)

   
   Compute various types of connectivity measures from EEG data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the time signal. Default is 250 Hz.
   :type sfreq: int
   :param kind: Type of connectivity measure to compute. The available options are:
                - For measures using Nilearn:
                  ["covariance", "correlation", "partial correlation", "tangent", "precision"]
                - For measures using MNE-connectivity:
                  ["ciplv", "ppc", "pli", "dpli", "wpli", "wpli2_debiased", "cohy", "imcoh", "coh", "plv", "gc", "gc_tr", "mic", "mim"]
                Where:
                  - "pli" stands for Phase Lag Index.
                  - "wpli" stands for weighted Phase Lag Index.
                  - "plv" stands for Phase Locking Value.
                  - "gc" stands for Granger Causality.
                  - "gc_tr" stands for Granger Causality with trends.
                  - "mic" stands for Maximal Information Coefficient.
                  - "mim" stands for Maximal Information Coefficient Mutual.
   :type kind: str
   :param filter_bank: Band-pass filter parameters with shape (2,) [low_freq, high_freq]. Default is None (no filtering).
   :type filter_bank: ndarray or list, optional
   :param n_win: Number of windows to split the data into. If the connectivity measure requires multiple epochs, this parameter helps in splitting one epoch into multiple parts. Default is 1.
   :type n_win: int

   :returns: Flattened array of the computed connectivity matrix with shape (n_channel * n_channel,).
   :rtype: ndarray

   .. rubric:: Notes

   - For certain measures like "tangent", multiple epochs are required. Ensure `n_win` is set appropriately for such measures.
   - If the `filter_bank` is specified, the data is band-pass filtered before computing the connectivity.
   - In case of an error during connectivity computation, the function returns an identity matrix and prints a warning message. Ensure the parameters are set correctly to avoid computation errors.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_cross_frequency_coupling(data, sfreq=250, band=np.array([[1, 4], [4, 8], [8, 10], [10, 13], [13, 20], [20, 30], [30, 45]]), mode='eeg_rhythm', low_fq_range=None, low_fq_width=2.0, high_fq_range='auto', high_fq_width='auto', method='tort', n_surrogates=0, n_jobs=1)

   
   Compute cross-frequency coupling using either 'eeg_rhythm' or 'Fixed_width' mode.

   :param data: The input data array with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the time signal. Default is 250 Hz.
   :type sfreq: int
   :param band: Frequency bands for analysis with shape (fea_num, 2). Default is predefined bands.
   :type band: ndarray
   :param mode: Mode of computation, either 'eeg_rhythm' or 'Fixed_width'. Default is 'eeg_rhythm'.
   :type mode: str
   :param low_fq_range: Filtering frequencies for phase signal in 'Fixed_width' mode. Default is None.
   :type low_fq_range: array or list
   :param low_fq_width: Bandwidth of the band-pass filter for phase signal. Default is 2.0.
   :type low_fq_width: float
   :param high_fq_range: Filtering frequencies for amplitude signal. Default is 'auto'.
   :type high_fq_range: array or list or 'auto'
   :param high_fq_width: Bandwidth of the band-pass filter for amplitude signal. Default is 'auto'.
   :type high_fq_width: float or 'auto'
   :param method: Method for computing modulation index. Default is 'tort'.
   :type method: str
   :param n_surrogates: Number of surrogates for z-score computation. Default is 0.
   :type n_surrogates: int
   :param n_jobs: Number of parallel jobs. Default is 1.
   :type n_jobs: int

   :returns: Cross-frequency coupling features with shape (n_channels, band_num, band_num) or (n_channels, low_fq_range.shape[0], high_fq_range.shape[0]).
   :rtype: ndarray

   Notes:
   - This function has been abandoned.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_detrended_fluctuation(data)

   
   Computes detrended fluctuation analysis for each channel.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray

   :returns: Detrended fluctuation per channel, shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_dispersion_entropy(data, classes=10, scale=1, emb_dim=2, delay=1, mapping_type='cdf', de_normalize=False, A=100, Mu=100, return_all=False, warns=True)

   
   :param data: ndarray,           shape (n_channels, n_times)
   :param classes: number of classes - (levels of quantization of amplitude) (default=10)
   :param emb_dim: embedding dimension,
   :param delay: time delay (default=1)
   :param scale: downsampled signal with low resolution  (default=1)  - for multipscale dispersion entropy
   :param mapping_type: mapping method to discretizing signal (default='cdf')
                        : options = {'cdf','a-law','mu-law','fd'}
   :param A: factor for A-Law- if mapping_type = 'a-law'
   :param Mu: factor for μ-Law- if mapping_type = 'mu-law'
   :param de_normalize: (bool) if to normalize the entropy, to make it comparable with different signal with different
                        number of classes and embeding dimensions. default=0 (False) - no normalizations
   :param if de_normalize=1:
                              - dispersion entropy is normalized by log(Npp); Npp=total possible patterns. This is classical
                                way to normalize entropy since   max{H(x)}<=np.log(N) for possible outcomes. However, in case of
                                limited length of signal (sequence), it would be not be possible to get all the possible patterns
                                and might be incorrect to normalize by log(Npp), when len(x)<Npp or len(x)<classes**emb_dim.
                                For example, given signal x with discretized length of 2048 samples, if classes=10 and emb_dim=4,
                                the number of possible patterns Npp = 10000, which can never be found in sequence length < 10000+4.
                                To fix this, the alternative way to nomalize is recommended as follow.
                              - select this when classes**emb_dim < (N-(emb_dim-1)*delay)

                             de_normalize=2: (recommended for classes**emb_dim > len(x)/scale)
                              - dispersion entropy is normalized by log(Npf); Npf [= (len(x)-(emb_dim - 1) * delay)]
                                the total  number of patterns founds in given sequence. This is much better normalizing factor.
                                In worst case (lack of better word) - for a very random signal, all Npf patterns could be different
                                and unique, achieving the maximum entropy and for a constant signal, all Npf will be same achieving to
                                zero entropy
                              - select this when classes**emb_dim > (N-(emb_dim-1)*delay)

                             de_normalize=3:
                              - dispersion entropy is normalized by log(Nup); number of total unique patterns (NOT RECOMMENDED)
                                -  it does not make sense (not to me, at least)

                             de_normalize=4:
                              - auto select normalizing factor
                              - if classes**emb_dim > (N-(emb_dim-1)*delay), then de_normalize=2
                              - if classes**emb_dim > (N-(emb_dim-1)*delay), then de_normalize=2

   Returns:         feature            shape (n_channels)















   ..
       !! processed by numpydoc !!

.. py:function:: compute_fuzzy_entropy(data, m=2, tau=1, r=(0.2, 2), Fx='default', Logx=np.exp(1))

   
   Compute fuzzy entropy for each channel in the data.

   :param data: The input data array with shape (n_channels, n_times).
   :type data: ndarray
   :param m: Embedding dimension. Default is 2.
   :type m: int
   :param tau: Time delay. Default is 1.
   :type tau: int
   :param r: Fuzzy function parameters. Default is (.2, 2).
   :type r: float or tuple
   :param Fx: Fuzzy function name. Default is 'default'.
   :type Fx: str
   :param Logx: Base of the logarithm for normalization. Default is e.
   :type Logx: float

   :returns: Fuzzy entropy features with shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_hosa_bicoherence(data, nfft=None, wind=None, nsamp=None, overlap=None)

   
   Compute the higher-order spectral analysis (HOSA) bicoherence of the data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param nfft: FFT length. Defaults to 128.
   :type nfft: int, optional
   :param wind: Time-domain window to be applied. Defaults to None, which uses a Hanning window.
   :type wind: array-like, optional
   :param nsamp: Samples per segment. Defaults to None.
   :type nsamp: int, optional
   :param overlap: Percentage overlap. Defaults to 50.
   :type overlap: float, optional

   :returns: Bicoherence features with shape (n_channels, nfft * nfft).
   :rtype: ndarray

   .. rubric:: Notes

   - This function is experimental and may have issues.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_multiscale_permutation_entropy(data, m=1, delay=1, scale=1)

   
   Compute the multiscale permutation entropy for each channel in the data.

   :param data: The input data array with shape (n_channels, n_times).
   :type data: ndarray
   :param m: Embedding dimension for permutation entropy. Default is 1.
   :type m: int
   :param delay: Time delay for permutation entropy. Default is 1.
   :type delay: int
   :param scale: Scale factor for multiscale permutation entropy. Default is 1.
   :type scale: int

   :returns: Multiscale permutation entropy features with shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_multiscale_sample_entropy(data, sample_length=1, tolerance=None, maxscale=None)

   
   Computes multiscale sample entropy for each channel.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray
   :param sample_length: Length of the sample.
   :type sample_length: int
   :param tolerance: Tolerance level for sample entropy.
   :type tolerance: float
   :param maxscale: Maximum scale for multiscale entropy.
   :type maxscale: int

   :returns: Multiscale sample entropy features, shape (n_channels, maxscale).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_offset_exponent_cf(data, sfreq=250, n=1024)

   
   Compute the offset and exponent of the power spectrum from EEG data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the signal. Default is 250 Hz.
   :type sfreq: int
   :param n: Number of frequency points for the Fourier transform. Typically set to the number of data points.
   :type n: int

   :returns: Flattened array of offset and exponent parameters with shape (n_channels * 2,).
   :rtype: ndarray

   .. rubric:: Notes

   - Computes the median frequency and the spectral slope (offset and exponent) using `compute_spect_slope`.
   - The spectral slope is inverted in the returned feature array.















   ..
       !! processed by numpydoc !!

.. py:function:: compute_pac_connectivity(data, sfreq=250, method='tort', band=np.array([[4, 8], [30, 45]]), n_surrogates=0, mode='self', approach_pac='mean')

   
   Compute Phase-Amplitude Coupling (PAC) connectivity from EEG data.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the time signal. Default is 250 Hz.
   :type sfreq: int
   :param method: Method for computing PAC. Options are:
                  - "tort": Tortoise method
                  - "jiang": Jiang method
   :type method: str
   :param band: Frequency bands for PAC computation with shape (2, 2). Each row specifies the low and high frequencies for the band.
   :type band: ndarray
   :param n_surrogates: Number of surrogates for significance testing. Default is 0 (no surrogates).
   :type n_surrogates: int
   :param mode: Mode for PAC computation. Options are:
                - "self": Compute PAC for each channel with itself.
                - "non-self": Compute PAC between each pair of channels.
   :type mode: str
   :param approach_pac: Approach for summarizing PAC values. Options are:
                        - "mean": Use the mean PAC value.
                        - "max": Use the maximum PAC value.
   :type approach_pac: str

   :returns:

             Flattened array of PAC connectivity features. The shape depends on the `mode`:
                 - If `mode` is "self": (n_channels,)
                 - If `mode` is "non-self": (n_channels * n_channels,)
   :rtype: ndarray

   .. rubric:: Notes

   - The `band` parameter specifies the frequency range for the low and high frequency bands used in PAC computation.
   - The `method` parameter determines the algorithm used for PAC calculation.
   - In "self" mode, PAC is computed for each channel individually.
   - In "non-self" mode, PAC is computed for every pair of channels.
   - The `approach_pac` parameter determines how the PAC values are aggregated: either by taking the mean or the maximum value.

   .. rubric:: Example

   To compute PAC using the "tort" method for each channel with itself, averaging the PAC values:
   ```python
   pac_features = compute_pac_connectivity(data, method='tort', mode='self', approach_pac='mean')
   ```















   ..
       !! processed by numpydoc !!

.. py:function:: compute_periodic_pac_connectivity(data, sfreq=250, n=1024, method='tort', band=np.array([[4, 8], [30, 45]]), n_surrogates=0, mode='self', approach_pac='mean')

   
   Compute periodic phase-amplitude coupling (PAC) connectivity from the signal data.

   :param data: Shape (n_channels, n_times). The input signal data.
   :type data: ndarray
   :param sfreq: The sampling frequency of the signal.
   :type sfreq: int
   :param n: The number of points for Fourier transform.
   :type n: int
   :param method: The method to use for PAC computation ('tort' or others).
   :type method: str
   :param band: An array specifying the frequency bands for PAC computation. Shape (2, 2).
   :type band: ndarray
   :param n_surrogates: The number of surrogate data to compute for significance testing.
   :type n_surrogates: int
   :param mode: The mode of PAC computation ('self' or others).
   :type mode: str
   :param approach_pac: The approach to compute PAC ('mean' or others).
   :type approach_pac: str

   :returns: The PAC connectivity feature. Flattened array of shape (n_channels * n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_perm_entropy(data)

   
   Computes permutation entropy for each channel.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray

   :returns: Permutation entropy per channel, shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_pow_freq_bands_cd(data, sfreq=250, freq_bands=np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 40]]), psd_method='welch', log=False)

   
   Compute the power in specified frequency bands from the power spectral density.

   :param data: Shape (n_channels, n_times). The input signal data.
   :type data: ndarray
   :param sfreq: The sampling frequency of the signal.
   :type sfreq: int
   :param freq_bands: An array of frequency bands for power computation. Shape (n_bands, 2).
   :type freq_bands: ndarray
   :param psd_method: The method to use for computing the power spectral density ('fft' or 'welch').
   :type psd_method: str
   :param log: Whether to apply a logarithm to the resulting power values.
   :type log: bool

   :returns: The power in each frequency band. Flattened array of shape (n_channels * n_bands,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_pow_freq_bands_remove_aperiodic(data, sfreq=250, freq_bands=np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 40]]), psd_method='welch', log=False, freq_range=None)

   
   Compute the power in specified frequency bands after removing aperiodic components.

   :param data: Shape (n_channels, n_times). The input signal data.
   :type data: ndarray
   :param sfreq: The sampling frequency of the signal.
   :type sfreq: int
   :param freq_bands: An array of frequency bands for power computation. Shape (n_bands, 2).
   :type freq_bands: ndarray
   :param psd_method: The method to use for computing the power spectral density ('fft' or 'welch').
   :type psd_method: str
   :param log: Whether to apply a logarithm to the resulting power values.
   :type log: bool
   :param freq_range: The frequency range for aperiodic component fitting. If None, no range is used.
   :type freq_range: tuple or None

   :returns: The power in each frequency band after removing aperiodic components. Flattened array of shape (n_channels * n_bands,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_stft_2019(data, sfreq=250, win_times=10, n_fre_idx=36)

   
   Compute the short-time Fourier transform (STFT) and sum the energy at fixed frequency points.

   :param data: The input data array with shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the time signal. Default is 250 Hz.
   :type sfreq: int
   :param win_times: Window length in seconds. Default is 10.
   :type win_times: int
   :param n_fre_idx: Number of frequency points to sum. Default is 36.
   :type n_fre_idx: int

   :returns: STFT-based features with shape (n_channels, section_num, n_fre_idx).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_test2(data)

   
   Computes the mean of the data along the last axis.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray

   :returns: Mean values, shape (n_channels,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: compute_wavelet_entropy(data, sfreq=250, m_times=1, m_Par_ratios=1, m_entropy=True, Average=True, wavelet_name='gaus1', band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]]))

   
   Computes wavelet entropy for given data.

   :param data: Input data of shape (n_channels, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency of the data.
   :type sfreq: float
   :param m_times: Time unit in seconds.
   :type m_times: float
   :param m_Par_ratios: Whether to include ratios (1) or not (0).
   :type m_Par_ratios: int
   :param m_entropy: Whether to use entropy formula (True) or energy formula (False).
   :type m_entropy: bool
   :param Average: Whether to compute average or not.
   :type Average: bool
   :param wavelet_name: Name of the wavelet to use.
   :type wavelet_name: str
   :param band: Frequency bands to analyze, shape (fea_num, 2).
   :type band: ndarray

   :returns:

             Computed features. If Average=True, shape is (n_channels, fea_num + m_Par_ratios * 2).
                      If Average=False, shape is (n_channels, (fea_num + m_Par_ratios * 2) * section_num).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: filter_bank(data, sfreq=250, frequences=None)

   
   Apply a filter bank to the input data.

   :param data: Input data with shape (n_channel, n_times).
   :type data: ndarray
   :param sfreq: Sampling frequency. Defaults to 250.
   :type sfreq: int, optional
   :param frequences: Frequency ranges for the filters. Shape (n_filters, 2). Defaults to None.
   :type frequences: ndarray, optional

   :returns: Filtered data with shape (n_filters, n_channel, n_times).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: find_nearest(array, value)

   
   Find the index of the nearest value in an array.

   :param array: Input array.
   :type array: array
   :param value: Value to find the nearest index for.
   :type value: float

   :returns: Index of the nearest value in the array.
   :rtype: int















   ..
       !! processed by numpydoc !!

.. py:function:: flatten_lower_triangle(matrix)

   
   Flatten the lower triangle of a square matrix into a 1D array.

   :param matrix: Square matrix to flatten.
   :type matrix: ndarray

   :returns: Flattened array with shape (n_channel*(n_channel-1)//2,).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_fft_values(y, N=None, f_s=250)

   
   Compute the FFT values of a time series.

   :param y: Input time series data.
   :type y: array
   :param N: Number of samples to use. Defaults to None.
   :type N: int, optional
   :param f_s: Sampling frequency. Defaults to 250.
   :type f_s: int, optional

   :returns: Frequencies and FFT values.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: get_power_from_channel(data, wind, windowsover, i_channel, channel, sfreq, freq1, freq2)

   
   Compute the power of a specified frequency band for a given channel.

   :param data: Input data with shape (n_channels, n_times).
   :type data: ndarray
   :param wind: Window length for the spectrogram.
   :type wind: int
   :param windowsover: Number of overlapping windows for the spectrogram.
   :type windowsover: int
   :param i_channel: Index of the channel to analyze.
   :type i_channel: int
   :param channel: List of channel names.
   :type channel: list
   :param sfreq: Sampling frequency of the signal.
   :type sfreq: int
   :param freq1: Lower bound of the frequency range.
   :type freq1: float
   :param freq2: Upper bound of the frequency range.
   :type freq2: float

   :returns: Power of the specified frequency band for the given channel.
   :rtype: ndarray

   .. rubric:: Notes

   - Uses `plt.specgram` to compute the power spectrum of the channel data.















   ..
       !! processed by numpydoc !!

.. py:function:: imp_extract_fft(section_data, Fs, time_sec)

   
   Extracts FFT from the data.

   :param section_data: Data segment to analyze.
   :type section_data: ndarray
   :param Fs: Sampling frequency.
   :type Fs: float
   :param time_sec: Length of the time segment.
   :type time_sec: int

   :returns: (m_fft, f), where m_fft is the FFT of the data, and f is the frequency vector.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: imp_extract_wavelet(section_data, Fs, time_sec, wavelet_name)

   
   Extracts wavelet transform from the data.

   :param section_data: Data segment to analyze.
   :type section_data: ndarray
   :param Fs: Sampling frequency.
   :type Fs: float
   :param time_sec: Length of the time segment.
   :type time_sec: int
   :param wavelet_name: Name of the wavelet to use.
   :type wavelet_name: str

   :returns: (cwt_re, f1), where cwt_re is the wavelet transform, and f1 is the frequency vector.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: pow_freq_bands_from_spectrum(freq_bands, freqs, spectrum_frequencies)

   
   Compute the power in specified frequency bands from the power spectral density.

   :param freq_bands: An array of frequency bands for power computation. Shape (n_bands, 2).
   :type freq_bands: ndarray
   :param freqs: Array of frequency values corresponding to the power spectrum.
   :type freqs: ndarray
   :param spectrum_frequencies: The power spectrum of the data. Shape (n_channels, n_freqs).
   :type spectrum_frequencies: ndarray

   :returns: The power in each frequency band. Shape (n_channels, n_bands).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: reshape_to_lower_triangle(flattened_array, n_channel)

   
   Reshape a 1D array into the lower triangle of a square matrix.

   :param flattened_array: 1D array to reshape.
   :type flattened_array: ndarray
   :param n_channel: Number of channels, defining the size of the square matrix.
   :type n_channel: int

   :returns: Square matrix with shape (n_channel, n_channel).
   :rtype: ndarray















   ..
       !! processed by numpydoc !!

