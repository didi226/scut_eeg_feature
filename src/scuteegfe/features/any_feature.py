import numpy as np
from scipy.signal import hilbert
from scipy.stats import iqr
from PyEMD import EMD
import antropy as ant
import pywt
from ..HOSA.conventional.bicoherence import bicoherence
from pyts.metrics.dtw import dtw
import antropy as ant

def compute_FDA(data, sfreq=250, win_times=1):
    """
    Detrended fluctuation analysis
    去趋势波动分析用于寻找时间序列中的长期统计相关性
    :param data: ndarray, shape (n_channels, n_times)
    :param data: win_times  窗口时间
    :return: ndarray, shape (n_channels, section_num*EMD_params*EMD_length)
    """
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=ant.detrended_fluctuation(data[i_channel, i_section * win_len:(i_section+ 1) * win_len])
    feature = feature.reshape(-1)
    return feature
def compute_Hilbert_abs(data):
    """
    希尔伯特模
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    n_channel, n_times = data.shape
    feature = abs(hilbert(data)).reshape(-1)
    # feature=feature[np.newaxis, :];
    return feature
def compute_EMD(data, sfreq=250, EMD_times=1, EMD_params=6):
    """
    经验模式分解
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels, section_num*EMD_params*EMD_length)
    """
    EMD_length = sfreq * EMD_times
    n_channel, n_times = data.shape
    section_num = n_times // EMD_length
    signal_imfs = np.zeros((n_channel, section_num, EMD_params, EMD_length))
    emd = EMD()
    for N_length in range(section_num):
        for N_channel in range(n_channel):
            IMFs = emd.emd(data[N_channel, N_length * EMD_length:(N_length + 1) * EMD_length])
            signal_imfs[N_channel, N_length, :, :] = IMFs[0:EMD_params, :]
    feature = signal_imfs.reshape(-1)
    return feature
def compute_hosa_bicoherence(data,nfft=None, wind=None, nsamp=None, overlap=None):
    '''
    :param data:    ndarray, shape (n_channels, n_times)
    :param nfft:    fft length [default = power of two > segsamp]
                    actual size used is power of two greater than 'nsamp'
                    默认 128
    :param wind:    specifies the time-domain window to be applied to each
                    data segment; should be of length 'segsamp' (see below);
                    otherwise, the default Hanning window is used.
    :param nsamp:   samples per segment [default: such that we have 8 segments]
                    if x is a matrix, segsamp is set to the number of rows
    :param overlap: percentage overlap, allowed range [0,99]. [default = 50];
                    if x is a matrix, overlap is set to 0.
    :return:        ndarray shape (n_channels, nfft*nfft)
    ex:
                    rng = np.random.RandomState(42)
                    n_epochs, n_channels, n_times = 2,2,250
                    X = rng.randn(n_epochs, n_channels, n_times)
                    feat=Feature(X,sfreq=250,selected_funcs={'hosa_bicoherence'})
                    bic=feat.features.reshape((n_epochs,n_channels,128,128))
    '''
    n_channel,n_times=data.shape
    feature=[]
    for N_channel in range(n_channel):
        y = data[N_channel, :]
        if y.ndim == 1:
            y = y[np.newaxis, :]
        bic, _ = bicoherence(y, nfft, wind, nsamp, overlap)
        feature.append(bic)
    feature = np.array(feature).reshape(-1)
    return feature
def compute_Itakura_Distance(data,baseline_data=None,dist='square', options={'max_slope': 2.0},
                                    precomputed_cost=None, return_cost=False,
                                    return_accumulated=False, return_path=False):
    """
    :reference                   https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html#pyts.metrics.dtw
    :param data:                 ndarray, shape (n_channels, n_times)
    :param baseline_data:        ndarray, shape (n_channels, n_times)
    :param dist:                 ‘square’, ‘absolute’, ‘precomputed’ or callable (default = ‘square’)
                                 Distance used. If ‘square’, the squared difference is used. If ‘absolute’,
                                 the absolute difference is used. If callable, it must be a function with a numba.njit() decorator
                                 that takes as input two numbers (two arguments) and returns a number.
                                 If ‘precomputed’, precomputed_cost must be the cost matrix and method must be ‘classic’,
                                 ‘sakoechiba’ or ‘itakura’.
    :param options:              None or dict (default = None)
                                 Dictionary of method options. Here is a quick summary of the available options for each method:
                                 ‘classic’: None
                                 ‘sakoechiba’: window_size (int or float)
                                 ‘itakura’: max_slope (float)
                                 ‘region’ : region (array-like)
                                 ‘multiscale’: resolution (int) and radius (int)
                                  ‘fast’: radius (int)
                                 For more information on these options, see show_options().
    :param precomputed_cost:     array-like, shape = (n_timestamps_1, n_timestamps_2) (default = None)
                                 Precomputed cost matrix between the time series. Ignored if dist != 'precomputed'.
    :param return_cost:          bool (default = False)
                                 If True, the cost matrix is returned.
    :param return_accumulated:   bool (default = False)
                                 If True, the accumulated cost matrix is returned.
    :param return_path:          bool (default = False)
                                 If True, the optimal path is returned.

    :return:                     ndarray shape (n_channels, ) Itakura_distance for every channel
    ex:                          rng = np.random.RandomState(42)
                                 n_epochs, n_channels, n_times = 2,2,2000
                                 rng_ = np.random.RandomState(43)
                                 data1 = rng.randn(n_epochs, n_channels, n_times)
                                 baseline_data=  np.squeeze(rng_.randn(1, n_channels, n_times))
                                 select_para=dict({'Itakura_Distance__baseline_data':baseline_data})
                                 feat=Feature(data1,selected_funcs={'Itakura_Distance'},funcs_params=select_para)
    """
    n_channel, n_times = data.shape
    Itakura_distance = np.zeros((n_channel))
    for i_channel in range(n_channel):
        Itakura_distance[i_channel] = dtw(x=data[i_channel, :], y=baseline_data[i_channel, :],
                                          dist=dist, method='itakura',
                                          options=options,
                                          precomputed_cost=precomputed_cost, return_cost=return_cost,
                                          return_accumulated=return_accumulated, return_path=return_path)
    return Itakura_distance




def compute_wavelet_entropy(data,sfreq=250,m_times=1,m_Par_ratios=1,m_entropy=True,Average=True,
                            wavelet_name= 'gaus1', band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]])):
    '''
    :param data:            ndarray, shape (n_channels, n_times)
    :param sfreq:           sfreq
    :param m_times:         time  uit  s
    :param m_Par_ratios:    ratios or not
                            1      ratios
                            0      no ratios
    :param m_entropy:       bool
                            True       m_entropy   公式含log
                            False      energy      公式不含log
    :param Average          bool
                            True       求平均
                            False      不求平均
    :param wavelet_name:    wavelet_name
    :param band:            ndarray shape (2,fea_num) [fre_low, frre_high]
    :return:                Average=True: ndarray shape (n_channels,fea_num+m_Par_ratios * 2)
                            Average=Flase:ndarray shape (n_channels,(fea_num+m_Par_ratios * 2)*section_num)
    :ex:
                            rng = np.random.RandomState(42)
                            n_epochs, n_channels, n_times = 2,2,2000
                            X = rng.randn(n_epochs, n_channels, n_times)
                            feat=Feature(X,sfreq=250,selected_funcs={'wavelet_entropy'})
    '''
    time_sec = int(m_times * sfreq)
    fea_num = int(band.shape[0] + m_Par_ratios * 2)
    n_channel, n_times = data.shape
    section_num = int(np.ceil((n_times / time_sec)))
    if Average:
        de = np.empty((int(n_channel), fea_num))
    else:
        de = np.empty((int(n_channel), fea_num, section_num))
    for channel in range(n_channel):
        # initialization
        section_de = np.empty((fea_num, section_num))
        # for one second calculate cwt
        for section in range(section_num):
            section_data = data[channel, section * time_sec:(section + 1) * time_sec]
            spec, f = imp_extract(section_data=section_data, Fs=sfreq, time_sec=time_sec, wavelet_name=wavelet_name)
            section_de[:, section] = band_DE(spec, f, Par_ratios=m_Par_ratios, band=band)
        if Average:
            de_mean = np.sum(section_de, axis=1);
        else:
            de_mean = section_de;
        if m_entropy:
            de_mean = np.multiply(de_mean, np.log(de_mean));
        de[channel, :] = de_mean
    feature = de.reshape(-1)
    return feature
def imp_extract(section_data,Fs, time_sec,wavelet_name):
    f = np.arange(1, 129, 0.2)
    [wt, f1] = pywt.cwt(section_data, f, wavelet_name, 1 / Fs)  # 'mexh'
    cwt_re = np.sum(abs(wt), axis=1) * 2 / time_sec;  #
    return cwt_re,f1
def band_DE(Pxx, f, Par_ratios=1, band=None):
    """
    Feature extraction of fixed frequency band
    :param Pxx:  frequency band parameter
    :param f:    frequency range
    :param band: selected frequency band
    :return:     固定频带的特征
    """
    fea_num=int(band.shape[0])
    psd = np.empty((fea_num))
    for i in range(fea_num):
        idx = np.where((f >= band[i, 0]) & (f <= band[i, 1]))
        psd[i] = np.sum(np.multiply(Pxx[idx], Pxx[idx]))
        if Par_ratios == 1:
            san_D = np.hstack((psd, psd[2] / psd[1], psd[3] / psd[1]))
        else:
            san_D = psd
    return san_D



def compute_test2(data):
    return np.mean(data, axis=-1)


def compute_IQR(data):
    """
    四分位数
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return iqr(data, axis=1)


def compute_Num_zero_crossings(data):
    """
    过零点数量
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return ant.num_zerocross(data, axis=1)

def compute_Petrosian_fd(data):
    """
    彼得罗森分形维数
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return ant.petrosian_fd(data, axis=1)
