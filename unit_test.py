import unittest
import numpy as np
import matplotlib.pyplot as plt
from scuteegfe import Feature

from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne_features.feature_extraction import extract_features
from scipy.signal import hilbert, coherence

def calculate_correlation_matrix(sample_data,sfreq, method):
    num_channels = sample_data.shape[0]
    correlation_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                channel_data1 = sample_data[i, :]
                channel_data2 = sample_data[j, :]
                if method == "correlation":
                    correlation_matrix[i, j] = calculate_channel_correlation_pearson(channel_data1, channel_data2)
                elif method == "plv":
                    correlation_matrix[i, j] = calculate_channel_correlation_plv(channel_data1, channel_data2)
                elif method == "coh":
                    correlation_matrix[i, j] = calculate_channel_correlation_coh(channel_data1, channel_data2,sfreq)
            else:
                correlation_matrix[i, j] = 1  # 自相关为1
    return correlation_matrix


def calculate_channel_correlation_pearson(channel_data1, channel_data2):
    pearson = (np.corrcoef(channel_data1, channel_data2))[0, 1]
    return np.abs(pearson)


def calculate_channel_correlation_plv(channel_data1, channel_data2):
    # 使用希尔伯特变换提取相位
    analytic_signal1 = hilbert(channel_data1)
    analytic_signal2 = hilbert(channel_data2)
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)
    # 计算 PLV
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv


def calculate_channel_correlation_coh(channel_data1, channel_data2,sfreq):
    # 计算每对信号之间的相干性
    _, coh = coherence(channel_data1, channel_data2, fs=sfreq, nperseg=channel_data1.shape[0] // 2)
    # 取相干性矩阵的平均值作为代表值
    mean_coh = np.mean(coh)
    return mean_coh
def get_data_example_motor_image():
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    subject = 1
    runs = [6, 10, 14]  # motor imagery: hands vs feet
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2
    data = epochs.get_data()
    print(data.shape)
    return data, labels

class MyTestCase(unittest.TestCase):

    def test_feature_name_order(self):
        # make fake data
        data = np.empty((10, 5, 100))
        for epoch in range(10):
            for ch in range(5):
                data[epoch, ch] = np.linspace(ch, ch + epoch + 1, 100)

        fea = Feature(data, selected_funcs=Feature.mne_defined_funcs)
        mean_index = np.argwhere(fea.feature_names == 'mean')
        fea_mean = fea.features[:, :, mean_index[0, 0]]
        true_mean = data.mean(axis=-1)
        self.assertTrue((fea_mean == true_mean).all())

    def test_visualize(self):
        data = np.empty((50, 5, 100))
        for epoch in range(50):
            for ch in range(5):
                data[epoch, ch] = np.linspace(ch, ch + epoch + 1, 100) + np.sin(np.linspace(-np.pi, np.pi, 100))

        data2 = np.empty((50, 5, 100))
        for epoch in range(50):
            for ch in range(5):
                data2[epoch, ch] = np.linspace(ch, ch + epoch + 1, 100) + np.random.randint(-100, 100) * 0.01 + np.sin(
                    np.linspace(-np.pi, np.pi, 100)) + 5
                if ch == 3:
                    data2[epoch, ch] = np.zeros(100)

        fea1 = Feature(data, selected_funcs=Feature.funcs_subset_no_spect_slope).reorder().fix_missing()
        fea2 = Feature(data2, selected_funcs=Feature.funcs_subset_no_spect_slope).reorder().fix_missing()

        Feature.plot_feature_sns(fea1, fea2, ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        plt.show()
        _, p = Feature.ttest_feature(fea1, fea2, ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        plt.show()
        print('Nan count: ', np.isnan(p).sum())
        self.assertTrue(True)

    def test_get_data(self):
        data, _ = get_data_example_motor_image()
        fea = Feature(data,sfreq=160, selected_funcs=['mean','std'])
        n_sample_list=np.arange(3,7)
        fea_part=fea.get_data(n_sample_list=n_sample_list)
        print(fea_part.shape)
        self.assertTrue((fea_part == fea.features[n_sample_list]).all())
        fea_part_0=fea.get_data(n_sample_list=1)
        self.assertTrue(fea_part_0.ndim==fea.features.ndim)


    def test_get_item(self):
        data,_=get_data_example_motor_image()
        #(45, 64, 801)
        fea = Feature(data, sfreq=160,selected_funcs=['mean','std'])
        # test int
        n_sample_list =1
        fea_new=fea[n_sample_list]
        print(fea_new.feature_names)
        self.assertTrue((fea_new.feature_names== fea.feature_names).all())
        self.assertTrue(fea_new.funcs == fea.funcs)
        self.assertTrue(fea_new.funcs_params == fea.funcs_params)
        print(fea_new.features.shape)
        print(fea.features.shape)
        self.assertTrue(1==fea_new.features.shape[0])
        self.assertTrue(fea.features.shape[1]== fea_new.features.shape[1])
        self.assertTrue(fea.features.shape[2]== fea_new.features.shape[2])
        #test slice
        fea_new_1=fea[4:10]
        self.assertTrue((fea_new_1.feature_names == fea.feature_names).all())
        self.assertTrue(fea_new_1.funcs == fea.funcs)
        self.assertTrue(fea_new_1.funcs_params == fea.funcs_params)
        print(fea_new_1.features.shape)
        self.assertTrue(6== fea_new_1.features.shape[0])
        self.assertTrue(fea.features.shape[1] == fea_new_1.features.shape[1])
        self.assertTrue(fea.features.shape[2] == fea_new_1.features.shape[2])
        #test non int
        fea_new_2 = fea[4.3]


    def test_Shannon_entropy_(self):
        data, _ = get_data_example_motor_image()
        # (45, 64, 801)
        fea = Feature(data, sfreq=160, selected_funcs=['Renyi_Entropy','Tsallis_Entropy','Shannon_entropy'],
                      funcs_params={"Renyi_Entropy__sfreq":160, "Tsallis_Entropy__sfreq":160,"Shannon_entropy__sfreq":160})
        print(fea.features)

    # def test_mne_feature_name(self):
    #     data, _ = get_data_example_motor_image()
    #     selec_fun = ['pow_freq_bands']
    #     select_para = dict({'pow_freq_bands__freq_bands': [[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]],
    #                         'pow_freq_bands__normalize': True,
    #                         'pow_freq_bands__ratios': 'all',
    #                         'pow_freq_bands__ratios_triu': True,
    #                         'pow_freq_bands__psd_method': 'fft',
    #                         'pow_freq_bands__log': True})
    #     fea=extract_features(data,sfreq=160,selected_funcs=selec_fun,funcs_params=select_para)
    def test_fuzzy_entropy(self):
        from scuteegfe import Feature
        data1 = np.random.rand(2, 3, 500)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['fuzzy_entropy'],
                       funcs_params={})
        print(fea1.features)

    def test_mne_coonectivty_plot(self):
        data1 = np.random.rand(1, 20, 500)
        from mne_connectivity import spectral_connectivity_epochs
        feature_1 = spectral_connectivity_epochs(data=data1, method='coh', mode='multitaper', sfreq=100,
                                                 fmin=2,
                                                 fmax=40, faverage=True, mt_adaptive=False).get_data()
        feature = feature_1.reshape((20, 20))
        feature = np.tril(feature, 0) + np.tril(feature, -1).T
    def test_feature_smooth(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data1 = np.random.rand(60, 3, 500)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['DFA'],
                       funcs_params={})
        print(fea1.features.shape)
        print(data1)
        feature1=fea1.features
        feature2 = fea1.feature_smooth(fea1.features,smooth_type="UnscentedKalmanFilter_sigmoid",window_size=3)
        #feature2 = fea1.feature_smooth(fea1.features, smooth_type="lds", window_size=3)
        print(feature2)
        print(fea1.features==feature2)
    def  test_compute_correlation_matrix(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        from nilearn.connectome import ConnectivityMeasure
        from nilearn import plotting
        import matplotlib.cm as cm
        ###构造随机矩阵
        data1 = np.random.rand(10, 20, 500)
        #针对第一维比较-直接计算
        data=data1[0]
        time_series = data.transpose(1, 0)
        time_series = time_series.reshape(1, int(time_series.shape[0]), time_series.shape[1])
        connectivity_measure = ConnectivityMeasure(kind="correlation")
        matrix_0 = connectivity_measure.fit_transform(time_series)[0]
        print(matrix_0)
        #比较-利用Feature类计算
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['correlation_matrix'],
                       funcs_params={"correlation_matrix__sfreq": 100,"correlation_matrix__kind":"coh"})

        print(fea1.features[0])
        ##计算相关性后可视化
        plotting.plot_matrix(fea1.features[0], vmin=-0.5, vmax=1, cmap=cm.jet, colorbar=False)
        plt.show()
        # 查看结果是不是一样
        self.assertTrue(fea1.features[0][5,4] == matrix_0[5,4])
    def  test_compute_correlation_dimension(self):
        data1 = np.random.rand(10, 20, 500)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['correlation_dimension'],
                       funcs_params={"correlation_dimension__emb_dim":10})
        print(fea1.features.shape)
        print(fea1.features)
    def  test_compute_dfa(self):
        data1 = np.random.rand(10, 20, 500)
        fea1 = Feature(data1, sfreq=100,selected_funcs=['DFA'],funcs_params={"DFA__sfreq":100}) #
        print(fea1.features.shape)
        print(fea1.features)

    def  test_compute_dispersion_entropy(self):
        data1 = np.random.rand(1, 20, 500)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['dispersion_entropy'])
        print(fea1.features.shape)
        print(fea1.features)
    def test_dft(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        from nilearn import plotting
        import matplotlib.cm as cm
        ###构造随机矩阵
        data1 = np.random.rand(10, 20, 500)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['correlation_matrix'],
                       funcs_params={"correlation_matrix__sfreq": 100, "correlation_matrix__kind":
                           "dtf","correlation_matrix__filter_bank": np.array([8,12])})
        print(fea1.features[0])
        ##计算相关性后可视化
        plotting.plot_matrix(fea1.features[0], vmin=0, vmax=0.1, cmap=cm.jet, colorbar=False)
        plt.show()
    def test_periodic_aperiodic_components(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        import scipy.io as sio
        ###构造随机矩阵
        mat_data = sio.loadmat(r"D:\files_save\Data\Pazhou-RS-45Hz\mat\stu_ss65_0701.mat")['eo_output'][:,:,:1000]
        fea1 = Feature(data=mat_data, sfreq=250,selected_funcs=['aperiodic_periodic_offset_exponent_cf'],
                       funcs_params={"aperiodic_periodic_offset_exponent_cf__n":512})
        print(fea1.features.shape)
        print(fea1.features[0,:,1])

    def test_offset_exponent_cf(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        ###构造随机矩阵
        data = np.random.rand(10, 20, 500)
        fea1 = Feature(data=data, sfreq=160,selected_funcs=['offset_exponent_cf'],
                funcs_params={"offset_exponent_cf__sfreq":160,
                       "offset_exponent_cf__n":512,
                                     "offset_exponent_cf__freq_range":[1,40]},
              )
        print(fea1.features.shape)
        print(fea1.features[1,:,1])

    def test_reative_power(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data1 = np.random.rand(10, 20, 500)
        fea1 = Feature(data=data1, sfreq=250,selected_funcs=['relative_power'])#,funcs_params={"relative_power__freq_bands": np.array([0.5,4,8,12]) }
        print(fea1.features)
        print(fea1.features.shape)
    def test_alpha(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data = np.random.rand(10, 26, 1000)
        fea1 = Feature(data=data, sfreq=100,selected_funcs=['alpha_asymetry'],funcs_params={"alpha_asymetry__mode": "eeglab"})
        #"definition_ln" "definition_ratio"  "eeglab" "definition_lnratio" "definition_ln_rel" "definition_ratio_rel" "definition_lnratio_rel"
        print(fea1.features)
        print(fea1.features.shape)
    def test_multi_feature(self):
        #测试特征 ['std','mean','teager_kaiser_energy']
        # ['spect_slope']
        # ['energy_freq_bands']
        # ['pow_freq_bands']
        #['wavelet_coef_energy']
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        from mne_features.feature_extraction import extract_features
        data = np.random.rand(2, 5, 1000)
        data_one_channnel = np.expand_dims(data[:,1,:], axis=1)
        print(data_one_channnel.shape)
        fea1 = Feature(data = data, sfreq=250, selected_funcs = ['DFA']).features
        fea2 = Feature(data = data_one_channnel, sfreq=250, selected_funcs =  ['DFA']).features
        fea3 = extract_features(X = data_one_channnel, sfreq = 250, selected_funcs =  ['DFA'])
        print(fea3.shape)
        print(fea1[:, 1,:].shape)
        print(fea2.shape)
        # Feature 和 extract_features单通道要一致  多个特征的时候这里可能排序不一样
        self.assertTrue(np.array_equal(fea2, np.expand_dims(fea3, axis=1)))
        # Feature 多通道和但单通道一致   pow_freq_bands这里的不一致可能是归一化问题
        self.assertTrue(np.allclose(np.expand_dims(fea1[:, 1,:], axis=1),fea2))

    def test_multi_feature_my_feature(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        from mne_features.feature_extraction import extract_features
        data = np.random.rand(2, 5, 1000)
        data_one_channnel = np.expand_dims(data[:, 1, :], axis=1)
        print(data_one_channnel.shape)
        fea1 = Feature(data=data, sfreq=250, selected_funcs=['DFA','fuzzy_entropy','multiscale_permutation_entropy',
                                            'Renyi_Entropy','Median_Frequency']).features
        fea2 = Feature(data=data_one_channnel, sfreq=250, selected_funcs=['DFA','fuzzy_entropy','multiscale_permutation_entropy',
                                             'Renyi_Entropy','Median_Frequency']).features

        print(fea1[:, 1, :].shape)
        print(fea2.shape)
        # Feature 多通道和但单通道一致   pow_freq_bands这里的不一致可能是归一化问题
        self.assertTrue(np.array_equal(np.expand_dims(fea1[:, 1, :], axis=1), fea2))
    def test_pow_freq_bands_remove_periodic(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data = np.random.rand(2, 10, 1000)
        fea1 = Feature(data=data, sfreq = 250, selected_funcs=['pow_freq_bands_remove_aperiodic'],
            funcs_params={"pow_freq_bands_remove_aperiodic__freq_bands": np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 40]]),
                          "pow_freq_bands_remove_aperiodic__freq_range":[1, 40],
                       "pow_freq_bands_remove_aperiodic__psd_method":'welch'}
                       )
        print(fea1.features.shape)
    def test_pac_connectivity(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data = np.random.rand(2, 5, 1000)
        fea1 = Feature(data = data, sfreq=250, selected_funcs=['pac_connectivity'],
                       funcs_params={
                                     "pac_connectivity__sfreq": 250,
                                     "pac_connectivity__method": 'jiang',
                                     "pac_connectivity__band": np.array([[4, 8], [30, 45]]),
                                     "pac_connectivity__mode": 'non-self'
                                     } )
        #fix_missing()
        #reorder()
        print(fea1.features.shape)
    def test_aac_connectivity(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        data = np.random.rand(3, 5, 2500)
        fea1 = Feature(data = data, sfreq=250, selected_funcs=['aac_connectivity'],
                       funcs_params={"aac_connectivity__sfreq": 250,
                                     "aac_connectivity__band": np.array([[1, 3], [30, 45]]),
                                     "aac_connectivity__mode": 'self'
                                     } )
        print(fea1.features.shape)
    def test_pec_connectivity(self):
        from mne_connectivity import envelope_correlation
        data = np.random.rand(1, 5, 1000)
        feature_1 = envelope_correlation(data,verbose=False)
        feature = np.squeeze(feature_1.get_data("dense"))
        print(feature.shape)

    def test_pec_connectivity_use(self):
        from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
        from nilearn import plotting
        import matplotlib.cm as cm
        data1 = np.random.rand(3, 5, 2001)
        fea1 = Feature(data1, sfreq=100,
                       selected_funcs=['correlation_matrix'],
                       funcs_params={'correlation_matrix__sfreq':100,"correlation_matrix__kind": 'coh',"correlation_matrix__n_win": 2})
        print(fea1.features[0])
        ##计算相关性后可视化
        plotting.plot_matrix(fea1.features[0], vmin=0, vmax=0.1, cmap=cm.jet, colorbar=False)
        plt.show()

    def test_plv_coh_corr_calculate(self):
        npz_data = np.load(r"F:\data\mhw\ASD001_fixed_window_10_step_None_fnirs_data_mhw.npz")['data'][0,:,:]
        feature_corr_0 = Feature(npz_data[np.newaxis,:,:], sfreq=10,
                       selected_funcs=['correlation_matrix'],
                       funcs_params={'correlation_matrix__sfreq':10,"correlation_matrix__kind":
                           'mcorrelation'})
        feature_corr__mat_0 = feature_corr_0.features[0,:,:]
        feature_corr__mat_1 = calculate_correlation_matrix(npz_data, sfreq=10,method='correlation')
        dd = 1







if __name__ == '__main__':
    import sys
    import os

    for p in sys.path:
        print(p)

    import pybispectra

    print(pybispectra.__file__)


    suite = unittest.TestSuite()
    suite.addTests(
        [MyTestCase('test_aac_connectivity')])  # test_net_eegnet_TR_crosssub  test_psd test_insub_classify
    runner = unittest.TextTestRunner()  # 通过unittest自带的TextTestRunner方法
    runner.run(suite)

