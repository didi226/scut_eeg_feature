from mne_features.feature_extraction import extract_features
import copy
from einops import rearrange
from ..features.any_feature import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import warnings
import numbers

class Feature:
    mne_defined_funcs = {'mean', 'variance', 'std', 'ptp_amp', 'skewness', 'kurtosis', 'rms', 'quantile',
                         'hurst_exp', 'app_entropy', 'samp_entropy', 'decorr_time', 'pow_freq_bands',
                         'hjorth_mobility_spect', 'hjorth_complexity_spect', 'hjorth_mobility',
                         'hjorth_complexity', 'higuchi_fd', 'katz_fd', 'zero_crossings', 'line_length',
                         'spect_slope', 'spect_entropy', 'svd_entropy', 'svd_fisher_info', 'energy_freq_bands',
                         'spect_edge_freq', 'wavelet_coef_energy', 'teager_kaiser_energy'}

    funcs_subset_no_spect_slope = {'mean', 'variance', 'std', 'ptp_amp', 'skewness', 'kurtosis', 'rms', 'quantile',
                                   'hurst_exp', 'app_entropy', 'samp_entropy', 'decorr_time', 'pow_freq_bands',
                                   'hjorth_mobility_spect', 'hjorth_complexity_spect', 'hjorth_mobility',
                                   'hjorth_complexity', 'higuchi_fd', 'katz_fd', 'zero_crossings', 'line_length',
                                   'spect_entropy', 'svd_entropy', 'svd_fisher_info', 'energy_freq_bands',
                                   'spect_edge_freq', 'wavelet_coef_energy', 'teager_kaiser_energy'}


    def __init__(self, data=None, sfreq=250, selected_funcs=None, funcs_params=None, n_jobs=1, ch_names=None,
                 return_as_df=False, log_teager_kaiser_energy=False):
        """
        Initialize the feature extractor.

        Args:
            data (ndarray, optional): Input data with shape (n_samples, n_channels, n_features). If None, only available features will be printed.
            sfreq (int, optional): Sampling frequency, default is 250 Hz.
            selected_funcs (list of str, optional): List of feature function names to compute. If None, default functions will be used.
            funcs_params (dict, optional): Parameters for feature functions. If None, default parameters will be used.
            n_jobs (int, optional): Number of processes to use, default is 1.
            ch_names (list of str, optional): List of channel names. If None, channel names will not be used.
            return_as_df (bool, optional): Whether to return features as a pandas.DataFrame. Default is False.
            log_teager_kaiser_energy (bool, optional): Whether to compute the logarithm of Teager-Kaiser energy. Default is False.

        Warns:
            UserWarning: Issued if sfreq is not equal to 250.

        Attributes:
            funcs (list of callable): List of feature functions to compute.
            feature_names_order (list of str): Order of feature names.
            example_data (ndarray): Example data with shape (1, 1).
            n_channel (int): Number of channels in the data.
            funcs_params (dict): Parameters for feature functions.
            __features_raw (ndarray): Raw computed features.
            __features (ndarray): Features rearranged for output.
            __features_fix (bool): Flag indicating if features have been fixed.
            __list_multi_feature (list of str): List of multi-feature names.

        """
        if data is None:
            print('available features:', self.mne_defined_funcs)
            self.__features = None
            return
        if sfreq!=250:
            warnings.warn("提案的函数sfreq需要再以参数的形式传入", UserWarning)
        self.__feature_names = None
        self.__features = None
        self.log_teager_kaiser_energy = log_teager_kaiser_energy
        funcs, feature_names_order = self.get_funcs(selected_funcs)
        self.funcs = funcs
        self.feature_names_order = feature_names_order
        self.example_data = data[0, 0][None, None]
        self.conn_example_data = data[0][None]
        self.n_channel = data.shape[1]
        self.funcs_params = funcs_params
        features = extract_features(data, sfreq, funcs, funcs_params, n_jobs, ch_names, return_as_df)
        if return_as_df:
            self.__features_raw = features

        self.__features_raw = features
        self.__features = rearrange(self.__features_raw, 'b (feature channel) -> b channel feature',
                                    channel=self.n_channel)
        self.__features_fix = False
        self.__list_multi_feature = ['teager_kaiser_energy0', 'spect_slope0',
                          'energy_freq_bands0', 'wavelet_coef_energy0', 'pow_freq_bands0']
    def __repr__(self):
        """
        Generate a string representation of the FeatureExtractor instance.

        If the features attribute is None, a prompt to input data is returned. Otherwise,
        the string representation includes the number of epochs, channels, and features,
        along with the feature names.

        Returns:
            str: String representation of the FeatureExtractor instance.
        """
        if self.features is None:
            return 'you should input the data'
        else:
            n_epochs, n_channels, n_features = self.features.shape
            return str(n_epochs) + '(epochs) x ' + str(n_channels) + '(channels) x ' + str(n_features) + '(features)' \
                   + '\nfeature names: ' + str(self.feature_names)

    def get_funcs(self, selected_funcs):
        """
         Retrieve custom feature decomposition functions based on selected function names.

         Args:
             selected_funcs (list of str or list of tuple): List of feature function names or tuples of
                 (function name, function reference) to retrieve.

         Returns:
             tuple: A tuple containing:
                 - funcs (set of tuple): A set of tuples where each tuple is (function name, function reference).
                 - feature_names_order (list of str): List of feature function names in the order they were provided.

         Raises:
             AttributeError: If the format of `selected_funcs` is not valid.

        """
        # 获取自定义的特征分解函数
        selected_funcs = list(selected_funcs)
        for i, each in enumerate(selected_funcs):
            if not {each}.issubset(self.mne_defined_funcs):
                selected_funcs[i] = (each, eval('compute_' + each))
        # 获取自定义的特征名
        feature_names_order = []
        funcs = set(selected_funcs)
        for each in funcs:
            if isinstance(each, tuple):
                f_name = each[0]
                assert isinstance(f_name, str)
                feature_names_order.append(each[0])
            elif isinstance(each, str):
                feature_names_order.append(each)
            else:
                raise AttributeError
        return funcs, feature_names_order

    def fix_missing(self):
        """
        Fix missing values in the features.

        This method uses the mean strategy to impute missing values. If the mean strategy fails,
        it falls back to using a constant strategy with a fill value of 0.

        Returns:
            FeatureExtractor: A new instance of FeatureExtractor with missing values imputed.

        """
        from sklearn.impute import SimpleImputer
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        feature = np.zeros_like(self.features)
        for i, each_epoch in enumerate(self.features):
            try:
                feature[i] = imp_mean.fit_transform(each_epoch)
            except Exception as e:
                print('Can not fix missing value using "mean" method, now try constant method ', e)
                imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
                feature[i] = imp_constant.fit_transform(each_epoch)

        n_F = copy.deepcopy(self)
        n_F.features = feature
        return n_F

    def reorder(self):
        """
        Reorder features alphabetically by their names.

        Returns:
            FeatureExtractor: A new instance of FeatureExtractor with features reordered alphabetically.
        """
        features = self.features
        feature_names = self.feature_names
        order = feature_names.argsort()
        n_F = copy.deepcopy(self)
        n_F.__features = features[:, :, order]
        n_F.__feature_names = feature_names[order]
        return n_F

    @property
    def features(self):
        """
        Get the features with possible multi-feature fixing.

        Returns:
            ndarray: The features array.
        """
        if self.__features_fix is True:
            return self.__features
        if np.any(np.isin(self.__list_multi_feature, self.feature_names)):
            self.fix_multi_feature(log = self.log_teager_kaiser_energy)
            self.__features_fix = True
        return self.__features

    @features.setter
    def features(self, features):
        """
        Set the features array.

        Args:
            features (ndarray): The new features array.
        """
        self.__features = features

    @property
    def feature_names(self):
        """
        Get the feature names.

        Returns:
            ndarray: Array of feature names.
        """
        if self.__feature_names is not None:
            return self.__feature_names
        feature_names = []
        feature_shapes = []
        for each_fea in self.feature_names_order:
            try:
                params = list(self.funcs_params.keys())
                matching_params = []
                for each_param in params:
                    match = each_param.startswith(each_fea)
                    if match:
                        matching_params.append(each_param)
                param_dict = {}
                for each_param in matching_params:
                    param = {each_param: self.funcs_params[each_param]}
                    param_dict.update(param)
            except Exception as e:
                param_dict = None
            if each_fea in ["pac_connectivity","correlation_matrix"]:
                fea_ = Feature(self.conn_example_data, selected_funcs={each_fea}, funcs_params=param_dict)
            else:
                fea_ = Feature(self.example_data, selected_funcs={each_fea}, funcs_params=param_dict)
            fea_shape = fea_.__features.shape[2]
            feature_names.append(each_fea)
            feature_shapes.append(fea_shape)

        feature_indexs = []
        for i, each_fea in enumerate(feature_names):
            if int(feature_shapes[i]) > 1:
                fea_sub_name = []
                for each in range(int(feature_shapes[i])):
                    fea_sub_name.append(each_fea + (str(each)))
                feature_indexs.extend(fea_sub_name)
            else:
                feature_indexs.extend([each_fea])
        self.__feature_names = np.array(feature_indexs)
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, f_names):
        """
        Get the feature names.

        Returns:
            ndarray: Array of feature names.
        """
        self.__feature_names = f_names

    def get_data(self,n_sample_list=None):
        """
       Get the feature vector array.

       Args:
           n_sample_list (list, optional): List of sample indices to slice the features array. Defaults to None.

       Returns:
           ndarray: The original or sliced features array.
       """
        if n_sample_list is None:
              fea= self.features
              return fea
        else:
            dim = self.features.ndim
            fea = self.features[n_sample_list,:,:]
            if fea.ndim<dim:
                fea=fea[None,:]
            return fea

    def copy(self):
        """
       Copy the instance of Feature.

       Returns:
           Feature: A copy of the object.
       """
        Feature = deepcopy(self)
        return Feature

    def __getitem__(self, item):
        """
        Slice the Feature instance on the sample dimension.

        Args:
            item (slice): Slice object for indexing the features array.

        Returns:
            Feature: A new instance of Feature with sliced features.
        """
        cls = type(self)
        Feature = self.copy()
        Feature.features = self.get_data(item)
        return Feature




    @staticmethod
    def plot_feature_sns(Feature1, Feature2, ch_names, sub_type1='type1', sub_type2='type2'):
        """
        Plot features using seaborn.

        Args:
            Feature1 (Feature): First Feature instance.
            Feature2 (Feature): Second Feature instance.
            ch_names (list): List of channel names.
            sub_type1 (str, optional): Description of the first dataset type. Defaults to 'type1'.
            sub_type2 (str, optional): Description of the second dataset type. Defaults to 'type2'.
        """
        fea1 = Feature1.features
        fea2 = Feature2.features

        # 归一化
        fea_concat = np.concatenate([fea1, fea2], axis=0)
        std = fea_concat.std(axis=0)
        mean = fea_concat.mean(axis=0)
        std[std == 0] = 1e-20
        mean[np.isnan(mean)] = 0
        fea_concat = (fea_concat - mean[None, :, :]) / (std[None, :, :])
        fea1 = fea_concat[:fea1.shape[0]]
        fea2 = fea_concat[fea1.shape[0]:]

        # 构造DataFrame
        df1 = Feature1.feature_df2plot(fea1, Feature1.feature_names, ch_names, sub_type=sub_type1)
        df2 = Feature2.feature_df2plot(fea2, Feature2.feature_names, ch_names, sub_type=sub_type2)
        df2plot = pd.concat([df1, df2])

        # 画图
        for ch in ch_names:
            plt.figure(figsize=[100, 10])
            sns.boxplot(data=df2plot.query('Channel=="' + ch + '"'), x='features', y='Value', hue='type')
            plt.title(ch)
            plt.show()

    @staticmethod
    def feature_df2plot(features, feature_names, ch_names, sub_type='sub_type_1'):
        """
        Convert features to a DataFrame for plotting.

        This method organizes features, feature names, and channel names into a pandas DataFrame,
        which can be used for visualization purposes, particularly with seaborn.

        Args:
            features (ndarray): The feature array with shape (n_samples, n_channels, n_features).
            feature_names (list): List of feature names.
            ch_names (list): List of channel names.
            sub_type (str, optional): Description of the dataset type. Defaults to 'sub_type_1'.

        Returns:
            pandas.DataFrame: DataFrame containing the features, channel names, type, and values,
                              structured for plotting.
        """
        from einops import repeat, rearrange
        fea_names = repeat(np.array(feature_names), 'n_f -> (n_epoch n_ch n_f)', n_epoch=features.shape[0],
                           n_ch=features.shape[1])
        ch_names = repeat(np.array(ch_names), 'n_ch -> (n_epoch n_ch n_f)', n_epoch=features.shape[0],
                          n_f=features.shape[2])
        values = rearrange(features, 'n_epoch n_ch n_f -> (n_epoch n_ch n_f)')

        return pd.DataFrame({
            'features': fea_names,
            'Channel': ch_names,
            'type': sub_type,
            'Value': values
        })

    @staticmethod
    def ttest_feature(Feature1, Feature2, ch_names):
        """
        Run t-test and visualize p-values using a heatmap.

        This function performs a t-test between the features of two Feature instances and visualizes the p-values
        using a heatmap. The heatmap shows the -log10 of the p-values for better visualization.

        Args:
            Feature1 (Feature): An instance of the Feature class.
            Feature2 (Feature): Another instance of the Feature class.
            ch_names (list): List of channel names.

        Returns:
            tuple: A tuple containing the t-statistics and p-values.
        """
        assert Feature1.features is not None and Feature2.features is not None
        from scipy import stats

        # TODO 这里默认Feature1和Feature2的特征顺序一致，如果不一致，将会有问题
        assert (Feature1.feature_names == Feature2.feature_names).all() == True, \
            print('feature not match:', Feature1.feature_names, Feature2.features)
        sta, p = stats.ttest_ind(Feature1.features, Feature2.features)
        log10_p = np.log10(p)
        thresh = np.log10(0.05)
        nan_index = np.isnan(p)
        if nan_index.sum() != 0:
            p[nan_index] = p[~nan_index].mean(axis=0)

        plt.figure(figsize=(16, 4))
        sns.heatmap(log10_p, square=True, center=thresh, cmap='coolwarm', vmin=-4, vmax=0,
                    yticklabels=ch_names, xticklabels=Feature1.feature_names)
        return sta, p
    def fix_multi_feature(self,log=True):
        """
            Fix multi-feature data by rearranging it.

            This function rearranges the multi-feature data to ensure proper alignment and, optionally,
            applies a logarithmic transformation to the 'teager_kaiser_energy' feature.

            Args:
                log (bool, optional): If True, applies a logarithmic transformation to the 'teager_kaiser_energy' feature.
                                      Defaults to True.

            Returns:
                None
        """
        present_features = [feature for feature in self.__list_multi_feature if feature in self.feature_names]
        for multi_feature_name in present_features:
            multi_feature_name = multi_feature_name [:-1]
            multi_feature_name_names = [multi_feature_name + str(i) for i in
                                          range(np.char.startswith(self.feature_names, multi_feature_name).sum())]

            get_index = lambda source, target: np.argwhere(source == target)[0, 0]
            reorder = lambda source, target: [get_index(each, target) for each in source]
            _rearrange_ = lambda features: rearrange(
                rearrange(features, 'n_sub n_ch n_fea-> n_sub (n_fea n_ch)'),
                'n_sub (n_ch n_fea)-> n_sub n_ch n_fea', n_ch=self.n_channel)
            ind = reorder(multi_feature_name_names, self.feature_names)
            if log and multi_feature_name =='teager_kaiser_energy':
                self.__features[:, :, ind] = np.log10(_rearrange_(self.__features[:, :, ind]))
            else:
                self.__features[:, :, ind] = _rearrange_(self.__features[:, :, ind])

    @staticmethod
    def moving_average_filter(data, window_size):
        """
        Apply a moving average filter to the data.

        This function applies a moving average filter with a specified window size to smooth the data.

        Args:
            data (array-like): The input data to be filtered.
            window_size (int): The size of the moving window.

        Returns:
            ndarray: The filtered data.
        """
        filtered_data = []
        window = [0] * window_size  # Initialize a window of size `window_size` with zeros
        for i in range(len(data)):
            window.pop(0)  # Remove the oldest data point from the window
            window.append(data[i])  # Add the current data point to the window
            filtered_data.append(sum(window) / window_size)  # Calculate the average and add it to the filtered data
        filtered_data = np.array(filtered_data)
        return filtered_data

    @staticmethod
    def lsd_KalmanFilter(data, window_size):
        """
        Apply a Kalman Filter for smoothing data.

        This function uses a Kalman Filter to smooth the input data within a specified window size.

        Args:
            data (array-like): The input data to be smoothed.
            window_size (int): The size of the processing window.

        Returns:
            ndarray: The smoothed data.
        """
        from pykalman import KalmanFilter
        window_num = data.shape[0] // window_size
        smoothed_feature = []
        for i_window in range(window_num + 1):
            begin_idx = window_size * i_window
            end_idx = window_size * (i_window + 1)
            if begin_idx >= data.shape[0]:
                continue
            if end_idx > data.shape[0]:
                end_idx = data.shape[0]
            data_window = data[begin_idx:end_idx]
            transition_covariance = np.diag([0.1, 0.1])
            transition_covariance = 0.1
            observation_covariance = 0.001
            initial_state_mean = np.mean(data_window)
            initial_state_covariance = 1
            kf = KalmanFilter(transition_covariance=transition_covariance,
                              observation_covariance=observation_covariance,
                              initial_state_mean=initial_state_mean,
                              initial_state_covariance=initial_state_covariance)
            # Estimate the parameters using the EM algorithm
            kf = kf.em(data_window)
            estimated_A = kf.transition_matrices
            estimated_C = kf.observation_matrices
            estimated_Q = kf.transition_covariance
            estimated_R = kf.observation_covariance

            smoothed_state_means, smoothed_state_covs = kf.smooth(data_window)
            smoothed_feature.extend(smoothed_state_means.flatten())
        smoothed_feature = np.array(smoothed_feature)
        return smoothed_feature

    @staticmethod
    def lsd_UnscentedKalmanFilter(data, window_size, observation_functions_type=None):
        """
        Apply an Unscented Kalman Filter for smoothing data.

        This function uses an Unscented Kalman Filter to smooth the input data within a specified window size.
        It supports optional sigmoid observation functions.

        Args:
            data (array-like): The input data to be smoothed.
            window_size (int): The size of the processing window.
            observation_functions_type (str, optional): The type of observation function. If 'sigmoid', applies a
                                                        sigmoid-based observation function. Defaults to None.

        Returns:
            ndarray: The smoothed data.
        """
        from pykalman import UnscentedKalmanFilter
        window_num = data.shape[0] // window_size
        smoothed_feature = []
        if observation_functions_type == "sigmoid":
            def measurement_function(x, w):
                return np.arctanh(x / (5 * 10 ** 7)) * 10 ** 7 + w

            def measurement_function_oly_x(x):
                return np.arctanh(x / (5 * 10 ** 7)) * 10 ** 7
        else:
            measurement_function = None
        for i_window in range(window_num + 1):
            begin_idx = window_size * i_window
            end_idx = window_size * (i_window + 1)
            if begin_idx >= data.shape[0]:
                continue
            if end_idx > data.shape[0]:
                end_idx = data.shape[0]
            data_window = data[begin_idx:end_idx]

            transition_covariance = 0.1
            observation_covariance = 0.001
            if observation_functions_type == "sigmoid":
                initial_state = [measurement_function_oly_x(x) for x in data_window]
                initial_state_mean = np.mean(initial_state)
            else:
                initial_state_mean = np.mean(data_window)
            initial_state_covariance = 1
            kf = UnscentedKalmanFilter(

                observation_functions=measurement_function,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance
            )

            smoothed_state_means, smoothed_state_covs = kf.smooth(data_window)
            smoothed_feature.extend(smoothed_state_means.flatten())
        smoothed_feature = np.array(smoothed_feature)
        return smoothed_feature

    def feature_smooth(self, data, smooth_type="lds", window_size=10):
        """
        Smooth features using specified smoothing techniques.

        This function smooths the features using various smoothing techniques, including moving average filter,
        linear dynamical system (LDS) approach, and Unscented Kalman Filter. The method and window size for
        smoothing can be specified.

        Args:
            data (ndarray): The input data with shape (n_epoch, n_channel, n_feature).
            smooth_type (str, optional): The type of smoothing technique to use. Options are:
                - "mv_av_filter": Moving average filter
                - "lds": Linear dynamic system (LDS) approach
                - "UnscentedKalmanFilter": Unscented Kalman Filter
                - "UnscentedKalmanFilter_sigmoid": Unscented Kalman Filter with sigmoid function
                Defaults to "lds".
            window_size (int, optional): The size of the processing window. Defaults to 10.

        Returns:
            ndarray: The smoothed data.

        References:
            Duan R N, Zhu J Y, Lu B L. Differential entropy feature for EEG-based emotion classification[C]//2013 6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013: 81-84.
            Shi L C, Lu B L. Off-line and on-line vigilance estimation based on linear dynamical system and manifold learning[C]//2010 Annual International Conference of the IEEE Engineering in Medicine and Biology. IEEE, 2010: 6587-6590.
            Zheng W L, Zhu J Y, Lu B L. Identifying stable patterns over time for emotion recognition from EEG[J]. IEEE Transactions on Affective Computing, 2017, 10(3): 417-429.
        """
        n_eopoch, n_channel, n_feature = data.shape
        print(n_eopoch, n_channel, n_feature)
        smoothed_data = np.zeros((n_eopoch, n_channel, n_feature))
        for i_feature in range(n_feature):
            for i_channel in range(n_channel):
                print(data[:, i_channel, i_feature].shape)
                if smooth_type == "mv_av_filter":
                    smoothed_data[:, i_channel, i_feature] = self.moving_average_filter(data[:, i_channel, i_feature],
                                                                                        window_size)
                if smooth_type == "lds":
                    smoothed_data[:, i_channel, i_feature] = self.lsd_KalmanFilter(data[:, i_channel, i_feature],
                                                                                   window_size)
                if smooth_type == "UnscentedKalmanFilter":
                    smoothed_data[:, i_channel, i_feature] = self.lsd_UnscentedKalmanFilter(
                        data[:, i_channel, i_feature],
                        window_size)
                if smooth_type == "UnscentedKalmanFilter_sigmoid":
                    smoothed_data[:, i_channel, i_feature] = self.lsd_UnscentedKalmanFilter(
                        data[:, i_channel, i_feature], window_size, "sigmoid")
        return smoothed_data




