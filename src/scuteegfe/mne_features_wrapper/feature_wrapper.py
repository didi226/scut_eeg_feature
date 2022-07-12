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

    def __init__(self, data=None, sfreq=250, selected_funcs=None, params=None, n_jobs=1, memory=None):
        if data is None:
            self.features = None
            self.feature_names = None
            return
        funcs, feature_names = self.get_funcs(selected_funcs)
        features = extract_features(data, sfreq, funcs, params, n_jobs, memory)
        self.features = rearrange(features, 'b (channel feature) -> b channel feature',
                                       channel=data.shape[1])
        self.feature_names = feature_names

    def __repr__(self):
        if self.features is None:
            return 'you should input the data'
        else:
            n_epochs, n_channels, n_features = self.features.shape
            return str(n_epochs)+'(epochs) x '+str(n_channels)+'(channels) x '+str(n_features)+'(features)'\
                    + '\nfeature names: ' + str(self.feature_names)

    def get_funcs(self, selected_funcs):
        # 获取自定义的特征分解函数
        selected_funcs = list(selected_funcs)
        for i, each in enumerate(selected_funcs):
            if not {each}.issubset(self.mne_defined_funcs):
                selected_funcs[i] = (each, eval('compute_' + each))
        # 获取自定义的特征名
        feature_names = []
        for each in selected_funcs:
            if isinstance(each, tuple):
                f_name = each[0]
                assert isinstance(f_name, str)
                feature_names.append(each[0])
            elif isinstance(each, str):
                feature_names.append(each)
            else:
                raise AttributeError
        return set(selected_funcs), feature_names
