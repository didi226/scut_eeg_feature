from mne_features.feature_extraction import extract_features
from einops import rearrange
from ..features.any_feature import *


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
                 return_as_df=False):
        """
        计算特征

        :param data: ndarray, (n_samples, n_channels, n_features)
        :param sfreq: 采样频率
        :param selected_funcs: 要计算的特征
        :param funcs_params: 参数
        :param n_jobs: 进程数
        :param ch_names: 通道名
        :param return_as_df: 以pandas.Dataframe格式输出
        """
        if data is None:
            print('available features:', self.mne_defined_funcs)
            self.features = None
            return
        funcs, feature_names_order = self.get_funcs(selected_funcs)
        self.feature_names_order = feature_names_order
        self.example_data = data[0, 0][None, None]
        self.funcs_params = funcs_params

        features = extract_features(data, sfreq, funcs, funcs_params, n_jobs, ch_names, return_as_df)
<<<<<<< HEAD
        self.features = rearrange(features, 'b (channel feature) -> b channel feature',
=======
        if return_as_df:
            self.features = features

        self.features = rearrange(features, 'b (feature channel) -> b channel feature',
>>>>>>> gitte_fe/master
                                  channel=data.shape[1])

    def __repr__(self):
        if self.features is None:
            return 'you should input the data'
        else:
            n_epochs, n_channels, n_features = self.features.shape
            return str(n_epochs) + '(epochs) x ' + str(n_channels) + '(channels) x ' + str(n_features) + '(features)' \
                   + '\nfeature names: ' + str(self.feature_names)

    def get_funcs(self, selected_funcs):
        # 获取自定义的特征分解函数
        selected_funcs = list(selected_funcs)
        for i, each in enumerate(selected_funcs):
            if not {each}.issubset(self.mne_defined_funcs):
                selected_funcs[i] = (each, eval('compute_' + each))
        # 获取自定义的特征名
        feature_names_order = []
        funcs = set(selected_funcs)
        for each in selected_funcs:
            if isinstance(each, tuple):
                f_name = each[0]
                assert isinstance(f_name, str)
                feature_names_order.append(each[0])
            elif isinstance(each, str):
                feature_names_order.append(each)
            else:
                raise AttributeError
        return funcs, feature_names_order

    @property
    def feature_names(self):
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
            fea_ = Feature(self.example_data, selected_funcs={each_fea}, funcs_params=param_dict)
            fea_shape = fea_.features.shape[2]
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
        return np.array(feature_indexs)
