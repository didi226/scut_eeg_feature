scuteegfe
=========

.. py:module:: scuteegfe

.. autoapi-nested-parse::

   
   scuteegfe public API.
















   ..
       !! processed by numpydoc !!


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


   .. py:property:: feature_names
      
      Get the feature names.

      :returns: Array of feature names.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:property:: features
      
      Get the features with possible multi-feature fixing.

      :returns: The features array.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: funcs_subset_no_spect_slope


   .. py:attribute:: mne_defined_funcs


