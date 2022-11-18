from mne_features.feature_extraction import extract_features
import copy
from einops import rearrange
from ..features.any_feature import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
            self.__features = None
            return
        self.__feature_names = None
        self.__features = None
        self.log_teager_kaiser_energy = log_teager_kaiser_energy
        funcs, feature_names_order = self.get_funcs(selected_funcs)
        self.funcs = funcs
        self.feature_names_order = feature_names_order
        self.example_data = data[0, 0][None, None]
        self.n_channel = data.shape[1]
        self.funcs_params = funcs_params

        features = extract_features(data, sfreq, funcs, funcs_params, n_jobs, ch_names, return_as_df)
        if return_as_df:
            self.__features_raw = features

        self.__features_raw = features
        self.__features = rearrange(self.__features_raw, 'b (feature channel) -> b channel feature',
                                    channel=self.n_channel)
        self.__features_fix = False

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
        修复异常值
        Returns
        -------
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
        按首字母排列特征
        Returns
        -------

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
        if self.__features_fix is True:
            return self.__features
        if np.isin('teager_kaiser_energy0', self.feature_names):
            self.fix_teager_kaiser_energy(log=self.log_teager_kaiser_energy)
            self.__features_fix = True
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

    @property
    def feature_names(self):
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
        self.__feature_names = f_names

    @staticmethod
    def plot_feature_sns(Feature1, Feature2, ch_names, sub_type1='type1', sub_type2='type2'):
        """

        Parameters
        ----------
        Feature1: Feature instance
        Feature2: Feature instance
        ch_names: list, 通道名
        sub_type1: 所属数据集类型描述
        sub_type2: 所属数据集类型描述

        Returns
        -------

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
        run ttest and visualize Pvalue using heatmap
        Parameters
        ----------
        Feature1: Feature instance
        Feature2: Feature instance
        ch_names: list, 通道名

        Returns
        -------

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

    def fix_teager_kaiser_energy(self, log=True):
        """
            mne_features 中，teager_kaiser_energy的特征排列方式其它特征相反，需修复
        Parameters
        ----------
        log: bool, 是否取对数

        Returns
        -------

        """
        teager_kaiser_energy_names = ['teager_kaiser_energy' + str(i) for i in
                                      range(np.char.startswith(self.feature_names, 'teager_kaiser_energy').sum())]

        get_index = lambda source, target: np.argwhere(source == target)[0, 0]
        reorder = lambda source, target: [get_index(each, target) for each in source]
        _rearrange_ = lambda features: rearrange(
            rearrange(features, 'n_sub n_ch n_fea-> n_sub (n_fea n_ch)'),
            'n_sub (n_ch n_fea)-> n_sub n_ch n_fea', n_ch=self.n_channel)

        ind = reorder(teager_kaiser_energy_names, self.feature_names)
        if log:
            self.__features[:, :, ind] = np.log10(_rearrange_(self.__features[:, :, ind]))
        else:
            self.__features[:, :, ind] = _rearrange_(self.__features[:, :, ind])
