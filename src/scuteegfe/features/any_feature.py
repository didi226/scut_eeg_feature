import mne.filter
import numpy as np
from scipy.signal import hilbert
from scipy.stats import iqr
from PyEMD import EMD
import antropy as ant
import pywt
from pyentrp import entropy as ent
from ..HOSA.conventional.bicoherence import bicoherence
from pyts.metrics.dtw import dtw
from scipy import signal
from scipy.fftpack import fft
import tftb
from statsmodels.tsa.arima.model import ARIMA
from pactools.comodulogram import Comodulogram
from EntropyHub import FuzzEn
from nilearn.connectome import ConnectivityMeasure

#from mne_features.bivariate import compute_spect_corr, compute_phase_lock_val



def compute_DFA(data, sfreq=250, win_times=1):
    """
    Detrended fluctuation analysis
    去趋势波动分析用于寻找时间序列中的长期统计相关性
    :param data: ndarray, shape (n_channels, n_times)
    :param data: win_times  窗口时间
    :return: ndarray, shape (n_channels, section_num)
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
def compute_Shannon_entropy(data, sfreq=250,round_para=1, win_times=1):
    """
    Shannon_entropy
    香农熵
    :param data: data        ndarray     shape (n_channels, n_times)
    :param data: win_times   int         窗口时间
    :param data: round_para  int         数据有效位数
    :return:     feature     ndarray     shape (n_channels, section_num)
    """
    data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=ent.shannon_entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len])
    feature = feature.reshape(-1)
    return feature
def Tsallis_Entropy(time_series,alpha=2):
    """
    Return the Tsallis_Entropy of the sample data.
    Args:
        time_series: Vector or string of the sample data
    Returns:
        The Tsallis_Entropy as float value
    reference：
        https://zhuanlan.zhihu.com/p/81462898
        这里有个疑问log的底数为啥是2
    """
    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)
    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq ** (alpha)
    ent =1/(1-alpha)*(ent-1)
    return ent

def Renyi_Entropy(time_series,alpha):
    """Return the Renyi_Entropy of the sample data.
    Args:
        time_series: Vector or string of the sample data
    Returns:
        The Renyi_Entropy as float value
    """
    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq**(alpha)
    ent=1/(1-alpha)*np.log2(ent)

    return ent
def compute_ARMA_kalman_filter(data,AR_p=10,MA_q=1):
    '''
    卡尔曼滤波器ARMA建模的系数
    :param data: ndarray, shape (n_channels, n_times)
    :param AR_p: AR模型的阶数
    :param MA_q: MA模型的阶数
    :return:　　　ndarray　shape (n_channel,AR_p+MA_q)
    ex:
    -----------
        rng = np.random.RandomState(42)
        n_epochs, n_channels, n_times = 2,1,1750
        X = rng.randn(n_epochs, n_channels, n_times)
        feat=Feature(X,sfreq=250,selected_funcs={'ARMA_kalman_filter'})#
        print(feat.features.shape)
    reference:
    -----------
    [1] Rossow A B, Salles E O T, Côco K F. Automatic sleep staging using a single-channel EEG modeling by Kalman filter
    and HMM[C]//ISSNIP Biosignals and Biorobotics Conference 2011. IEEE, 2011: 1-6.
    '''
    n_channel, n_times = data.shape
    feature=np.zeros((n_channel,AR_p+MA_q))
    for i_channel in range(n_channel):
        arma_mod = ARIMA(data[i_channel,:], order=(AR_p, 0, MA_q))
        arma_res = arma_mod.fit()
        feature[i_channel,:]=np.concatenate([arma_res.polynomial_ar[1:], arma_res. polynomial_ma[1:]])
    feature = feature.reshape(-1)
    return feature

def get_fft_values(y, N=None, f_s=250):
    '''
    :param y:   array  times
    :param N:   使用的样本数
    :param f_s: 采样频率
    :return:
    f_values    频谱对应的频率
    fft_values  频谱
    '''
    if N is None:
        N=y.shape[0]
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft(y)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def find_nearest(array, value):
    '''
    :param array:  array
    :param array:  array
    :return: 返回array中最靠近value的位置坐标
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def compute_Harmonic_Parameters(data,sfreq=250,
                                band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]])):
   '''
   谐波参数（English：  Harmonic Parameters）
   波参数包括三个参数：  中心频率 (fc) 、带宽 (fr) 和中心频率处的频谱值 (Sfc)
   :param data:      ndarray, shape (n_channels, n_times)
   :param sfreq:     采样频率
   :param band:      对应的频带   中心频率 (fc),带宽 (fr)
   :return:          ndarray   shape (n_channels,n_band)
                     中心频率处的频谱值 (Sfc)
   '''
   n_channel, n_times = data.shape
   band_num=band.shape[0]
   feature=np.zeros((n_channel,band_num))
   for i_channel in range(n_channel):
     f_, fft_ = get_fft_values(data[i_channel, :], f_s=sfreq)
     for i_band in range(band_num):
           center_frequency=(band[i_band,0] + band[i_band,1]) / 2
           frequency_band=np.abs(band[i_band,0] - band[i_band,1])
         #  feature[i_channel,i_band,0]=center_frequency
         #  feature[i_channel, i_band, 1] = frequency_band
           f_idx=find_nearest(f_,center_frequency)
           feature[i_channel, i_band]=fft_[f_idx]
   feature = feature.reshape(-1)
   return feature
def compute_Median_Frequency(data,sfreq=250,
                                band=np.array([[0.5,2],[2, 4], [4, 5],
                                [5, 7], [7, 10], [10, 13],[13,15],[15,20],[20,30],[30,40]])):
    '''

    :param data:    ndarray, shape (n_channels, n_times)
    :param sfreq:   采样频率
    :param band:    对应频带
    :return:        ndarray, shape (n_channels, band_num)
    reference:
    ---------------
                   [1] Gudmundsson S, Runarsson T P, Sigurdsson S. Automatic sleep staging using support vector machines with posterior probability estimates
                       [C]//International Conference on Computational Intelligence for Modelling, Control and Automation and International Conference on Intelligent Agents,
                       Web Technologies and Internet Commerce (CIMCA-IAWTIC'06). IEEE, 2005, 2: 366-372.
                       （默认band参考论文）
                   [2] Thongpanja S, Phinyomark A, Phukpattaranont P, et al. Mean and median frequency of EMG signal to determine muscle force based on
                       time-dependent power spectrum[J]. Elektronika ir Elektrotechnika, 2013, 19(3): 51-56.
                       (Median_Frequency定义参考）
    '''
    n_channel, n_times = data.shape
    band_num = band.shape[0]
    feature = np.zeros((n_channel, band_num))
    for i_channel in range(n_channel):
       f_, fft_ = get_fft_values(data[i_channel, :], f_s=sfreq)
       feature[i_channel,:]=band_Median_Frequency(Pxx=fft_,f=f_,band=band)
    feature = feature.reshape(-1)
    return feature
def band_Median_Frequency(Pxx, f, band=None):
    """
     Feature extraction of fixed frequency band
    :param Pxx:  frequency band parameter
    :param f:    frequency range
    :param band: selected frequency band
    :return:     Median_Frequency
    """
    fea_num=int(band.shape[0])
    Median_Frequency=np.empty((fea_num));
    for i in range(fea_num):
        psd_m = 0
        idx = np.where((f >= band[i, 0]) & (f <= band[i, 1]))[0]
        psd= np.sum(Pxx[idx]**2)
        psd_half=psd/2
        Median_Frequency[i] = band[i, 0]

        for i_idx in idx:
            if(psd_m < psd_half):
                psd_m=psd_m+np.multiply(Pxx[i_idx], Pxx[i_idx])
            else:
                Median_Frequency[i]=f[i_idx]
                break

    return Median_Frequency

def filter_bank(data,sfreq=250,frequences=None):
    '''
    :param data:               [n_channel,n_times]
    :param frequences:         [n_filters,2]  low_frequence high_frequence
    :return:filters_data:      [n_filters,n_channel,n_times]
    '''
    n_filters=frequences.shape[0]
    n_channel,n_times=data.shape
    filters_data=np.zeros((n_filters,n_channel,n_times))
    for i_filters in range(n_filters):
        b, a = signal.butter(8, [2*frequences[i_filters,0]/sfreq, 2*frequences[i_filters,1]/sfreq], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
        filters_data[i_filters, :] = signal.filtfilt(b, a, data)  # data为要过滤的信号
    return filters_data

def compute_Coherence(data,Co_channel=None,
            sfreq=250,band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]])):
    """
    相干性反映了来自不同导数的两个信号在某些频域中的线性相关程度。
    Automatic sleep scoring: A search for an optimal combination of measures
    :param data:          ndarray, shape (n_channels, n_times)
    :param Co_channel:    ndarray shape [n_Co_channel,2] 需要计算相关性的通道序号
    :param sfreq:         sfreq
    :param band:          ndarray shape (fea_num,2) [fre_low, frre_high]
    :return:              feature  ndarray shape    (n_channel, n_channel * band_num) 未计算相关性部分数值为0
    """
    n_channel, n_times = data.shape
    band_num=band.shape[0]
    feature = np.zeros((n_channel, n_channel ,band_num))
    if Co_channel is None:
        Co_channel=np.zeros((n_channel*n_channel,2),dtype=np.int32)
        ij_channel=0
        for i_channel in range(n_channel):
            for j_channel in range(n_channel):
                Co_channel[ij_channel,:]=[i_channel,j_channel]
                ij_channel=ij_channel+1

    for i_Co_channel in range(Co_channel.shape[0]):
           channel_0=Co_channel[i_Co_channel,0];  channel_1=Co_channel[i_Co_channel,1];
           x=data[channel_0,:];     y=data[channel_1,:]
           print(x.shape,y.shape)
           ff,cxx=signal.coherence(x, y, fs=sfreq,
                            window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=- 1)
           feature[channel_0,channel_1,:]=band_DE(cxx,ff,Par_ratios=0,band=band)
    feature = feature.reshape(-1)
    return feature
def  compute_WignerVilleDistribution(data,sfreq=250 ):
    '''
    reference:Şen B, Peker M, Çavuşoğlu A, Çelebi FV. A comparative study on classification of sleep stage based
    on EEG signals using feature selection and classification algorithms.
    J Med Syst. 2014 Mar;38(3):18. doi: 10.1007/s10916-014-0018-0. Epub 2014 Mar 9. PMID: 24609509.
    :param data: ndarray, shape (n_channels, n_times)
    :param sfreq:
    :return:     ndarray, shape (n_channels, 4)
    这里对于最大频率的理解存疑
    '''
    n_channel, n_times = data.shape
    feature = np.zeros((n_channel, 4))
    for i_channel in range(n_channel):
        wvd=tftb.processing.WignerVilleDistribution(signal=data[i_channel,:],
                                                timestamps=np.arange(n_times)*(1/sfreq))
        tfr_wvd, t_wvd, f_wvd = wvd.run()
        feature[i_channel,:]=np.polyfit(t_wvd, tfr_wvd[-1,:], 3)
    feature = feature.reshape(-1)
    return feature

def compute_Renyi_Entropy(data, sfreq=250,round_para=1, win_times=1,alpha=2):
    """
    Renyi_Entropy
    tallis熵是Shannon(或Boltzmann-Gibbs)熵在熵非扩展情况下的推广
    :param data: data        ndarray     shape (n_channels, n_times)
    :param data: win_times   int         窗口时间
    :param data: round_para  int         数据有效位数
    :return:     feature     ndarray     shape (n_channels, section_num*EMD_params*EMD_length)
    """
    data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=Renyi_Entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len],alpha=alpha)
    feature = feature.reshape(-1)
    return feature
def compute_Tsallis_Entropy(data, sfreq=250,round_para=1, win_times=1,alpha=2):
    """
    Tsallis_Entropy
    tallis熵是Shannon(或Boltzmann-Gibbs)熵在熵非扩展情况下的推广
    :param data:data          ndarray     shape (n_channels, n_times)
    :param data:win_times     int         窗口时间
    :param data: round_para   int         数据有效位数
    :return:feature           ndarray     shape (n_channels, section_num*EMD_params*EMD_length)
    """
    data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=Tsallis_Entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len],alpha=alpha)
    feature = feature.reshape(-1)
    return feature


def compute_Hilbert_abs(data):
    """
    希尔伯特包络
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
    -------------
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
    ex:
    ------------
                                 rng = np.random.RandomState(42)
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
    :param band:            ndarray shape (fea_num,2) [fre_low, frre_high]
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
            spec, f = imp_extract_wavelet(section_data=section_data, Fs=sfreq, time_sec=time_sec, wavelet_name=wavelet_name)
            section_de[:, section] = band_DE(spec, f, Par_ratios=m_Par_ratios, band=band)
        section_de=Processing_inf_nan(section_de)
        if Average:
            de_mean = np.sum(section_de, axis=1);
        else:
            de_mean = section_de;
        if m_entropy:
            de_mean[de_mean == 0] = 1e-6
            de_mean = np.multiply(de_mean, np.log(de_mean));
        de[channel, :] = de_mean
    feature = de.reshape(-1)
    return feature
def imp_extract_wavelet(section_data,Fs, time_sec,wavelet_name):
    f = np.arange(1, 129, 0.2)
    [wt, f1] = pywt.cwt(section_data, f, wavelet_name, 1 / Fs)  # 'mexh'
    cwt_re = np.sum(abs(wt), axis=1) * 2 / time_sec;  #
    return cwt_re,f1
def imp_extract_fft(section_data,Fs,time_sec):
        f = np.arange(time_sec) / (time_sec / Fs)
        m_fft = abs(fft(section_data, time_sec) * 2 / time_sec);
        return m_fft[range(int(time_sec / 2))], f[range(int(time_sec/ 2))]
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
        if psd[1]==0:
            psd[1]=1e-6
        san_D = np.hstack((psd, psd[2] / psd[1], psd[3] / psd[1]))
    else:
        san_D = psd
    return san_D

def Processing_inf_nan(data):
    data_inf = np.isinf(data)
    data[data_inf] = 0
    data_nan = np.isnan(data)
    data[data_nan] = np.max(data)
    return data

def compute_test2(data):
    return np.mean(data, axis=-1)



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


def compute_perm_entropy(data):
    """
    排列熵
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return np.array([ant.perm_entropy(each_channel, normalize=True) for each_channel in data])


def compute_detrended_fluctuation(data):
    """
    去趋势波动分析法
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return np.array([ant.detrended_fluctuation(each_channel) for each_channel in data])


def compute_multiscale_entropy(data, sample_length=1, tolerance=None, maxscale=None):
    """
    多尺度熵
    :param maxscale:
    :param tolerance:
    :param sample_length:
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels, n_timepoints)
    """
    return np.array(
        [ent.multiscale_entropy
         (each_channel, sample_length=sample_length, tolerance=tolerance, maxscale=maxscale) for each_channel in data])


def compute_multiscale_permutation_entropy(data, m=1, delay=1, scale=1):
    """
    多尺度熵
    :param data: ndarray, shape (n_channels, n_times)
    :param m:
    :param delay:
    :param scale:
    :return: ndarray, shape (n_channels,)
    """
    return np.array([
        ent.multiscale_permutation_entropy(each_channel, m, delay, scale) for each_channel in data]).reshape(-1)


def compute_fuzzy_entropy(data,m=2, tau=1, r=(.2,2), Fx='default', Logx=np.exp(1)):
    """
    Args:
        data:  ndarray, shape (n_channels, n_times)
        fuzzy_entropy 计算官方文档地址：https://www.entropyhub.xyz/python/Functions/Base.html
        m:Embedding Dimension, a positive integer [default: 2]
        嵌入维度，一个正整数[默认值：2]
        tau:Time Delay, a positive integer [default: 1]
        时间延迟，一个正整数[默认值：1]
        Fx:Fuzzy function name, one of the following strings: {'sigmoid', 'modsampen', 'default', 'gudermannian', 'linear'}
         模糊函数名，以下字符串之一：{ 'sigmoid' , 'modsampen' , 'default' , 'gudermannian' , 'linear' }
        r:Fuzzy function parameters, a 1 element scalar or a 2 element vector of positive values. The r parameters for each fuzzy
        模糊函数参数，一个 1 元素标量或一个 2 元素正值向量。每个模糊的 r 参数
    Returns: (n_channel,)

    """
    n_channel, n_times = data.shape
    FuzzEn_value = np.zeros((n_channel,m))
    for i_channel in  range(n_channel):
        FuzzEn_value[i_channel],_,_=FuzzEn(data[i_channel],m=m,tau=tau,Fx=Fx,r=r,Logx=Logx)
    feature = FuzzEn_value.reshape(-1)
    return feature





def compute_cross_frequency_coupling(data,sfreq=250,band=np.array([[1,4], [4,8],[8,10], [10,13], [13,20], [20,30], [30,45]]),
                 mode='eeg_rhythm', low_fq_range=None, low_fq_width=2., high_fq_range='auto',
                 high_fq_width='auto', method='tort', n_surrogates=0,n_jobs=1):
    """
    Args:
        data:                    ndarray, shape (n_channels, n_times)
        sfreq:                   freq of time signal
        band:                    ndarray shape (fea_num,2) [fre_low, frre_high]      带通滤波器组参数
        mode:                    'eeg_rhythm'   计算EEG对应节律频率耦合
                                 ’Fixed_width‘  计算频带长度固定的频率耦合
        ’Fixed_width‘ 情况下有意义的参数：
                                    low_fq_range:    array or list
                                                     List of filtering frequencies (phase signal)
                                    low_fq_width:    float
                                                     Bandwidth of the band-pass filter (phase signal)
                                    high_fq_range:   array or list or 'auto'
                                                     List of filtering frequencies (amplitude signal)
                                                     If 'auto', it uses np.linspace(max(low_fq_range), fs / 2.0, 40).
                                    high_fq_width:   float or 'auto'
                                                     Bandwidth of the band-pass filter (amplitude signal)
                                                      If 'auto', it uses 2 * max(low_fq_range).
        method:              string or DAR instance
                              Modulation index method:

                            - String in ('ozkurt', 'canolty', 'tort', 'penny', ), for a PAC
                                estimation based on filtering and using the Hilbert transform.
                            - String in ('vanwijk', ) for a joint AAC and PAC estimation
                                based on filtering and using the Hilbert transform.
                            - String in ('sigl', 'nagashima', 'hagihira', 'bispectrum', ), for
                                a PAC estimation based on the bicoherence.
                            - String in ('colgin', ) for a PAC estimation
                                and in ('jiang', ) for a PAC directionality estimation,
                                based on filtering and computing coherence.
                            - String in ('duprelatour', ) or a DAR instance, for a PAC estimation
                                based on a driven autoregressive model.
        n_surrogates:   int
                        Number of surrogates computed for the z-score
                        If n_surrogates <= 1, the z-score is not computed.
        n_jobs:         Number of jobs to use in parallel computations.
                        Recquires scikit-learn installed.
    Returns:            feature：  ndarray shape (n_channel,band_num,band_num)
                                   or  (n_channel,low_fq_range.shape[0],high_fq_range.shape[0])
    ex:
    """
    n_channel, n_times = data.shape
    if mode=='eeg_rhythm':
        band_num = band.shape[0]
        feature = np.zeros((n_channel, band_num, band_num))
        for i_band in range(band_num):
            i_center_fq = (band[i_band, 0] + band[i_band, 1]) / 2
            i_fq_width = band[i_band, 1] - band[i_band, 0]
            for j_band in range(band_num):
                    j_center_fq = (band[j_band, 0] + band[j_band, 1]) / 2
                    j_fq_width = band[j_band, 1] - band[j_band, 0]
                    c = Comodulogram(fs=sfreq, low_fq_range=np.array(i_center_fq), low_fq_width=i_fq_width,
                                     high_fq_range=np.array(j_center_fq),high_fq_width=j_fq_width,method=method, n_surrogates=n_surrogates,n_jobs=n_jobs)
                    for N_channel in range(n_channel):
                      sig=data[N_channel,:]
                      feature[N_channel,i_band,j_band]=c.fit(sig).comod_
        feature = feature.reshape(-1)
    if mode=='Fixed_width':
        feature_=np.zeros((n_channel,low_fq_range.shape[0],high_fq_range.shape[0]))
        c = Comodulogram(fs=sfreq, low_fq_range=low_fq_range, low_fq_width=low_fq_width,
                         high_fq_range=high_fq_range, high_fq_width=high_fq_width,method=method, n_surrogates=n_surrogates,n_jobs=n_jobs)
        for N_channel in range(n_channel):
            sig = data[N_channel, :]
            feature_[N_channel,:,:]=c.fit(sig).comod_
        feature=feature_.reshape(-1)
    return feature


def  compute_stft_2019(data,sfreq=250,win_times=10,n_fre_idx=36):
    """
    Args:
        利用stft时频变换将固定频率点的能量求和作为特征
        参考文献：Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods
        data:             ndarray, shape (n_channels, n_times)
        sfreq:            sfreq
        win_times:        win_times 窗口时间
        n_fre_idx:        需要求和的频率点

    Returns:
    """
    from scipy.signal import stft
    win_len = sfreq * win_times
    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num,n_fre_idx))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            X= data[i_channel, i_section * win_len:(i_section + 1) * win_len]
            f, t, Zxx = stft(X, fs=sfreq, window='blackman', nperseg=256, noverlap=None, nfft=512,
                             detrend=False, boundary='zeros', padded=True)
            Y=10*np.log(abs(np.mean(Zxx[2:n_fre_idx*2+1:2,:],axis=1)))
            feature[i_channel, i_section] =Y
    feature = feature.reshape(-1)
    return feature


def flatten_lower_triangle(matrix):
    rows = len(matrix)
    flattened = []
    for i in range(rows):
        for j in range(i):
            flattened.append(matrix[i][j])
    flattened=np.array(flattened)
    return flattened

def reshape_to_lower_triangle(flattened_array,n_channel):
    matrix=np.zeros((n_channel,n_channel))
    count = 0
    for i in range(n_channel):
        for j in range(i):
            matrix[i][j] = flattened_array[count]
            count += 1
    return matrix
def compute_correlation_matrix(data,sfreq=250,kind="correlation",filter_bank=None,n_win=1):
    """
    Parameters
    ----------
    data               ndarray,           shape (n_channels, n_times)
    sfreq              int                freq of time signal
    kind               str                计算相关矩阵的类型
                                          利用nilearn的库计算
                                          ["covariance","correlation", "partial correlation", "tangent","precision",
                                          利用mne-connectivity计算
                                          "ciplv", "ppc", "pli", "dpli", "wpli", "wpli2_debiased", "cohy", "imcoh","coh","plv"]
    filter_bank        ndarray or list,   shape (2,) [fre_low, frre_high]      带通滤波器组参数 默认None即不做滤波
    n_win              int                窗口的个数,部分的特征无法利用单一的epoch计算功能连接，因此选择将一个epoch切成可以计算的形状
                                          例如：shape(30,2500)-->shape(2,30,1250)

    Returns
    -------
    shape              int                (n_channel*n_channel)
    note:                                 存在data再次进入的情况,即调用的时候n个epoch计算这个函数n次,但是会以错误的data的形状进入(n+1)次，
                                          try避免这种影响,保证最后获得的特征无误,但是容易因为参数传的错误找不到报错
                                          当发现计算错误的时候，主要检查n_win与计算的功能连接特征类型是不是匹配
    """

    n_channel,n_times=data.shape
    #tangent 这个是多个epoch放在一起才能计算的
    if kind in ["covariance","correlation", "partial correlation", "tangent","precision"]:
        if filter_bank is not None:
            data = mne.filter.filter_data(data, sfreq=sfreq, l_freq=filter_bank[0], h_freq=filter_bank[1])
        time_series = data.transpose(1, 0)
        time_series = time_series.reshape(n_win, n_times//n_win,n_channel)
        connectivity_measure = ConnectivityMeasure(kind=kind)
        matrix_0 = connectivity_measure.fit_transform(time_series)[0]
        feature = matrix_0
    elif kind in ['ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased', 'cohy', 'imcoh','coh','plv']:
        ##['coh', 'plv'] 'cohy', 'imcoh'无法使用
        new_data=data.reshape([n_win,n_channel,n_times//n_win])
            ###这部分计算就可以进行结果也有问题
        from mne_connectivity import spectral_connectivity_epochs
        try:
            if filter_bank is None:
                feature_1 = spectral_connectivity_epochs(data=new_data, method=kind, mode='multitaper', sfreq=sfreq,
                                                         faverage=True, mt_adaptive=False)
            else:
                feature_1=spectral_connectivity_epochs(data=new_data,method=kind,mode='multitaper', sfreq=sfreq, fmin=filter_bank[0],
                                                       fmax=filter_bank[1],faverage=True, mt_adaptive=False)
                # feature = feature_1.reshape((n_channel,n_channel))
                # feature = np.tril(feature, 0) + np.tril(feature, -1).T
            feature_1 = np.squeeze(feature_1.get_data("dense"))
            np.fill_diagonal(feature_1,1)
            feature = feature_1 + feature_1.T - np.diag(feature_1.diagonal())
        except:
               feature=np.eye(n_channel)
               print("feature connectivity jump")
    feature = feature.reshape(-1)
    return feature