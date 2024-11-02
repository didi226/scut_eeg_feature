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
from statsmodels.tsa.arima.model import ARIMA
from pactools.comodulogram import Comodulogram
#from pactools.utils import rayleigh_z
from EntropyHub import FuzzEn
from nilearn.connectome import ConnectivityMeasure
from .pdc_dtf import calculate_dtf_pdc
from fooof import FOOOFGroup
from scipy.stats import norm
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
import matplotlib.pyplot as plt
from mne_features.univariate import compute_pow_freq_bands
from scipy.signal import welch
from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
# import tftb





def compute_DFA(data, sfreq=250, win_times=1):
    """
    Perform Detrended Fluctuation Analysis (DFA) to find long-term statistical correlations in a time series.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        win_times (int, optional): Window duration in seconds. Defaults to 1.

    Returns:
        ndarray: DFA features with shape (n_channels, section_num).

    References:
        Peng C K, Havlin S, Stanley H E, Goldberger A L. Quantification of scaling exponents and crossover phenomena in
        nonstationary heartbeat time series[J]. Chaos: An Interdisciplinary Journal of Nonlinear Science, 1995, 5(1): 82-87.
    """
    win_len = sfreq * win_times
    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section] = ant.detrended_fluctuation(data[i_channel, i_section * win_len:(i_section+ 1) * win_len])
    feature = feature.T.reshape(-1)
    return feature
def compute_Shannon_entropy(data, sfreq=250,round_para=None, win_times=1):
    """
    Compute the Shannon entropy of the data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        round_para (int, optional): Rounding precision for data. Defaults to None, default retention of all digits for calculation.
        win_times (int, optional): Window duration in seconds. Defaults to 1.

    Returns:
        ndarray: Shannon entropy features with shape (n_channels, section_num).

    References:
        Shannon C E. A mathematical theory of communication[J]. Bell System Technical Journal, 1948, 27(3): 379-423.
    """
    if round_para is not None:
        data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=ent.shannon_entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len])
    feature = feature.T.reshape(-1)
    return feature
def Tsallis_Entropy(time_series,alpha=2):
    """
    Compute the Tsallis entropy of the sample data.

    Args:
        time_series (Union[Vector, str]): Input time series data.
        alpha (float, optional): Entropy parameter. Defaults to 2.

    Returns:
        float: Tsallis entropy value.

    References:
        Tsallis C. Possible generalization of Boltzmann-Gibbs statistics[J]. Journal of Statistical Physics, 1988, 52(1-2): 479-487.
        https://zhuanlan.zhihu.com/p/81462898 (Chinese reference explaining the base of the logarithm used)

    Note:
        There is a question about why the base of the logarithm used is 2.
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
    """
     Compute the Renyi entropy of the sample data.

     Args:
         time_series (Union[Vector, str]): Input time series data.
         alpha (float): Entropy parameter.

     Returns:
         float: Renyi entropy value.
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
    """
    Compute ARMA modeling coefficients using the Kalman filter.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        AR_p (int, optional): Order of the AR model. Defaults to 10.
        MA_q (int, optional): Order of the MA model. Defaults to 1.

    Returns:
        ndarray: ARMA coefficients with shape (n_channel, AR_p + MA_q).

    Example:
        >>> rng = np.random.RandomState(42)
        >>> n_epochs, n_channels, n_times = 2, 1, 1750
        >>> X = rng.randn(n_epochs, n_channels, n_times)
        >>> feat = Feature(X, sfreq=250, selected_funcs={'ARMA_kalman_filter'})
        >>> print(feat.features.shape)

    References:
        [1] Rossow A B, Salles E O T, Côco K F. Automatic sleep staging using a single-channel EEG modeling by Kalman filter
        and HMM[C]//ISSNIP Biosignals and Biorobotics Conference 2011. IEEE, 2011: 1-6.
    """
    n_channel, n_times = data.shape
    feature=np.zeros((n_channel,AR_p+MA_q))
    for i_channel in range(n_channel):
        arma_mod = ARIMA(data[i_channel,:], order=(AR_p, 0, MA_q))
        arma_res = arma_mod.fit()
        feature[i_channel,:]=np.concatenate([arma_res.polynomial_ar[1:], arma_res. polynomial_ma[1:]])
    feature = feature.T.reshape(-1)
    return feature

def get_fft_values(y, N=None, f_s=250):
    """
    Compute the FFT values of a time series.

    Args:
        y (array): Input time series data.
        N (int, optional): Number of samples to use. Defaults to None.
        f_s (int, optional): Sampling frequency. Defaults to 250.

    Returns:
        tuple: Frequencies and FFT values.

    """
    if N is None:
        N=y.shape[0]
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft(y)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.

    Args:
        array (array): Input array.
        value (float): Value to find the nearest index for.

    Returns:
        int: Index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def compute_Harmonic_Parameters(data,sfreq=250,
                                band=np.array([[2, 3.8], [4, 7], [8, 13], [14, 30], [31, 48]])):
   """
    Compute harmonic parameters including center frequency, bandwidth, and spectral value at the center frequency.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        band (ndarray, optional): Frequency bands with shape (n_bands, 2). Defaults to predefined bands.

    Returns:
        ndarray: Harmonic parameters with shape (n_channels, n_band).
    """
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
   feature = feature.T.reshape(-1)
   return feature
def compute_Median_Frequency(data,sfreq=250,
                                band=np.array([[0.5,2],[2, 4], [4, 5],
                                [5, 7], [7, 10], [10, 13],[13,15],[15,20],[20,30],[30,40]]),N=None):
    """
    Compute the median frequency for each channel and frequency band.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        band (ndarray, optional): Frequency bands. Defaults to predefined bands.
        N (int, optional): Number of samples to use. Defaults to None.

    Returns:
        ndarray: Median frequency values with shape (n_channels, band_num).

    References:
        [1] Gudmundsson S, Runarsson T P, Sigurdsson S. Automatic sleep staging using support vector machines with posterior probability estimates
            [C]//International Conference on Computational Intelligence for Modelling, Control and Automation and International Conference on Intelligent Agents,
            Web Technologies and Internet Commerce (CIMCA-IAWTIC'06). IEEE, 2005, 2: 366-372.
        [2] Thongpanja S, Phinyomark A, Phukpattaranont P, et al. Mean and median frequency of EMG signal to determine muscle force based on
            time-dependent power spectrum[J]. Elektronika ir Elektrotechnika, 2013, 19(3): 51-56.
    """
    n_channel, n_times = data.shape
    band_num = band.shape[0]
    feature = np.zeros((n_channel, band_num))
    for i_channel in range(n_channel):
       f_, fft_ = get_fft_values(data[i_channel, :],N=N, f_s=sfreq)
       feature[i_channel,:]=band_Median_Frequency(Pxx=fft_,f=f_,band=band)
    feature = feature.T.reshape(-1)
    return feature
def band_Median_Frequency(Pxx, f, band=None):
    """
    Compute the median frequency for a given power spectral density (PSD) and frequency band.

    Args:
        Pxx (array): Power spectral density values.
        f (array): Frequency values.
        band (ndarray, optional): Selected frequency bands. Defaults to None.

    Returns:
        ndarray: Median frequency values for each band.
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
    """
    Apply a filter bank to the input data.

    Args:
        data (ndarray): Input data with shape (n_channel, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        frequences (ndarray, optional): Frequency ranges for the filters. Shape (n_filters, 2). Defaults to None.

    Returns:
        ndarray: Filtered data with shape (n_filters, n_channel, n_times).
    """
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
    Compute coherence between different channels within specified frequency bands.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        Co_channel (ndarray, optional): Channels to compute coherence for. Shape (n_Co_channel, 2). Defaults to None.
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        band (ndarray, optional): Frequency bands. Shape (fea_num, 2). Defaults to predefined bands.

    Returns:
        ndarray: Coherence features with shape (n_channel, n_channel * band_num).

    Notes:
        For single-channel data, this function is not applicable.
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
    feature = feature.T.reshape(-1)
    return feature
# def  compute_WignerVilleDistribution(data,sfreq=250 ):
#     '''
#     reference:Şen B, Peker M, Çavuşoğlu A, Çelebi FV. A comparative study on classification of sleep stage based
#     on EEG signals using feature selection and classification algorithms.
#     J Med Syst. 2014 Mar;38(3):18. doi: 10.1007/s10916-014-0018-0. Epub 2014 Mar 9. PMID: 24609509.
#     :param data: ndarray, shape (n_channels, n_times)
#     :param sfreq:
#     :return:     ndarray, shape (n_channels, 4)
#     这里对于最大频率的理解存疑
#     '''
#     n_channel, n_times = data.shape
#     feature = np.zeros((n_channel, 4))
#     for i_channel in range(n_channel):
#         wvd=tftb.processing.WignerVilleDistribution(signal=data[i_channel,:],
#                                                 timestamps=np.arange(n_times)*(1/sfreq))
#         tfr_wvd, t_wvd, f_wvd = wvd.run()
#         feature[i_channel,:]=np.polyfit(t_wvd, tfr_wvd[-1,:], 3)
#     feature = feature.T.reshape(-1)
#     return feature

def compute_Renyi_Entropy(data, sfreq=250,round_para=None, win_times=1,alpha=2):
    """
     Compute the Renyi entropy for each channel using a sliding window approach.

     Args:
         data (ndarray): Input data with shape (n_channels, n_times).
         sfreq (int, optional): Sampling frequency. Defaults to 250.
         round_para (int, optional): Number of decimal places to round the data.  Defaults to None, default retention of all digits for calculation.
         win_times (int, optional): Window duration in seconds. Defaults to 1.
         alpha (float, optional): Renyi entropy parameter. Defaults to 2.

     Returns:
         ndarray: Computed Renyi entropy with shape (n_channels, section_num).

     Notes:
         - The entropy is calculated for each window of data and then averaged across all windows.
     """
    if round_para is not None:
        data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=Renyi_Entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len],alpha=alpha)
    feature = feature.T.reshape(-1)
    return feature
def compute_Tsallis_Entropy(data, sfreq=250,round_para=None, win_times=1,alpha=2):
    """
    Compute the Tsallis entropy for each channel using a sliding window approach.

    Args:
      data (ndarray): Input data with shape (n_channels, n_times).
      sfreq (int, optional): Sampling frequency. Defaults to 250.
      round_para (int, optional): Number of decimal places to round the data.  Defaults to None, default retention of all digits for calculation.
      win_times (int, optional): Window duration in seconds. Defaults to 1.
      alpha (float, optional): Tsallis entropy parameter. Defaults to 2.

    Returns:
      ndarray: Computed Tsallis entropy with shape (n_channels, section_num).

    Notes:
      - The entropy is calculated for each window of data and then averaged across all windows.
    """
    if round_para is not None:
        data = np.round(data, round_para)
    win_len = sfreq * win_times

    n_channel, n_times = data.shape
    section_num = n_times // win_len
    feature = np.zeros((n_channel, section_num))
    for i_section in range(section_num):
        for i_channel in range(n_channel):
            feature[i_channel,i_section]=Tsallis_Entropy(data[i_channel, i_section * win_len:(i_section+ 1) * win_len],alpha=alpha)
    feature = feature.T.reshape(-1)
    return feature


def compute_Hilbert_abs(data):
    """
    Compute the absolute value of the Hilbert transform (envelope) of the data. (abandon)

    Args:
     data (ndarray): Input data with shape (n_channels, n_times).

    Returns:
     ndarray: Absolute value of the Hilbert transform with shape (n_channels,).

    Notes:
     - This function is currently deprecated.
    """
    n_channel, n_times = data.shape
    feature = abs(hilbert(data)).reshape(-1)
    # feature=feature[np.newaxis, :];
    return feature
def compute_EMD(data, sfreq=250, EMD_times=1, EMD_params=6):
    """
    Compute the Empirical Mode Decomposition (EMD) of the data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int, optional): Sampling frequency. Defaults to 250.
        EMD_times (int, optional): Duration of each EMD segment in seconds. Defaults to 1.
        EMD_params (int, optional): Number of Intrinsic Mode Functions (IMFs) to extract. Defaults to 6.

    Returns:
        ndarray: EMD features with shape (n_channels, section_num * EMD_params * EMD_length).

    Notes:
        - The EMD is applied to segments of the data, and the resulting IMFs are used as features.
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
    signal_imfs = signal_imfs.reshape((n_channel, -1))
    feature = signal_imfs.T.reshape(-1)
    return feature
def compute_hosa_bicoherence(data,nfft=None, wind=None, nsamp=None, overlap=None):
    """
    Compute the higher-order spectral analysis (HOSA) bicoherence of the data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        nfft (int, optional): FFT length. Defaults to 128.
        wind (array-like, optional): Time-domain window to be applied. Defaults to None, which uses a Hanning window.
        nsamp (int, optional): Samples per segment. Defaults to None.
        overlap (float, optional): Percentage overlap. Defaults to 50.

    Returns:
        ndarray: Bicoherence features with shape (n_channels, nfft * nfft).

    Notes:
        - This function is experimental and may have issues.
    """
    n_channel,n_times=data.shape
    feature=[]
    for N_channel in range(n_channel):
        y = data[N_channel, :]
        if y.ndim == 1:
            y = y[np.newaxis, :]
        bic, _ = bicoherence(y, nfft, wind, nsamp, overlap)
        feature.append(bic)
    feature = np.array(feature).reshape((n_channel,-1))
    feature = feature.T.reshape(-1)
    return feature
def compute_Itakura_Distance(data,baseline_data = None,dist='square', options={'max_slope': 2.0},
                                    precomputed_cost=None, return_cost=False,
                                    return_accumulated=False, return_path=False):
    """
    Compute the Itakura distance between the data and baseline data using dynamic time warping (DTW).

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        baseline_data (ndarray, optional): Baseline data with shape (n_channels, n_times). Defaults to None.
        dist (str or callable, optional): Distance metric to use. Defaults to 'square'.
        options (dict, optional): Method options. Defaults to {'max_slope': 2.0}.
        precomputed_cost (array-like, optional): Precomputed cost matrix. Defaults to None.
        return_cost (bool, optional): If True, return the cost matrix. Defaults to False.
        return_accumulated (bool, optional): If True, return the accumulated cost matrix. Defaults to False.
        return_path (bool, optional): If True, return the optimal path. Defaults to False.

    Returns:
        ndarray: Itakura distance with shape (n_channels,).

    References:
        https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html#pyts.metrics.dtw
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
    """
    Computes wavelet entropy for given data.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).
        sfreq (float): Sampling frequency of the data.
        m_times (float): Time unit in seconds.
        m_Par_ratios (int): Whether to include ratios (1) or not (0).
        m_entropy (bool): Whether to use entropy formula (True) or energy formula (False).
        Average (bool): Whether to compute average or not.
        wavelet_name (str): Name of the wavelet to use.
        band (ndarray): Frequency bands to analyze, shape (fea_num, 2).

    Returns:
        ndarray: Computed features. If Average=True, shape is (n_channels, fea_num + m_Par_ratios * 2).
                 If Average=False, shape is (n_channels, (fea_num + m_Par_ratios * 2) * section_num).
    """
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
            de_mean = np.sum(section_de, axis=1)
        else:
            de_mean = section_de
        if m_entropy:
            de_mean[de_mean == 0] = 1e-6
            de_mean = np.multiply(de_mean, np.log(de_mean))
        de[channel, :] = de_mean
    feature = de.T.reshape(-1)
    return feature
def imp_extract_wavelet(section_data,Fs, time_sec,wavelet_name):
    """
    Extracts wavelet transform from the data.

    Args:
        section_data (ndarray): Data segment to analyze.
        Fs (float): Sampling frequency.
        time_sec (int): Length of the time segment.
        wavelet_name (str): Name of the wavelet to use.

    Returns:
        tuple: (cwt_re, f1), where cwt_re is the wavelet transform, and f1 is the frequency vector.
    """
    f = np.arange(1, 129, 0.2)
    [wt, f1] = pywt.cwt(section_data, f, wavelet_name, 1 / Fs)  # 'mexh'
    cwt_re = np.sum(abs(wt), axis=1) * 2 / time_sec;  #
    return cwt_re,f1
def imp_extract_fft(section_data,Fs,time_sec):
    """
    Extracts FFT from the data.

    Args:
        section_data (ndarray): Data segment to analyze.
        Fs (float): Sampling frequency.
        time_sec (int): Length of the time segment.

    Returns:
        tuple: (m_fft, f), where m_fft is the FFT of the data, and f is the frequency vector.
    """
    f = np.arange(time_sec) / (time_sec / Fs)
    m_fft = abs(fft(section_data, time_sec) * 2 / time_sec);
    return m_fft[range(int(time_sec / 2))], f[range(int(time_sec/ 2))]
def band_DE(Pxx, f, Par_ratios=1, band=None):
    """
    Computes features from fixed frequency bands.

    Args:
        Pxx (ndarray): Power spectral density.
        f (ndarray): Frequency vector.
        Par_ratios (int): Whether to compute ratios (1) or not (0).
        band (ndarray): Frequency bands to analyze, shape (fea_num, 2).

    Returns:
        ndarray: Computed features.
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
    """
    Processes infinite and NaN values in the data.

    Args:
        data (ndarray): Input data to process.

    Returns:
        ndarray: Processed data with infinite and NaN values handled.
    """
    data_inf = np.isinf(data)
    data[data_inf] = 0
    data_nan = np.isnan(data)
    data[data_nan] = np.max(data)
    return data

def compute_test2(data):
    """
    Computes the mean of the data along the last axis.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).

    Returns:
        ndarray: Mean values, shape (n_channels,).
    """
    return np.mean(data, axis=-1)



def compute_Num_zero_crossings(data):
    """
    Computes the number of zero crossings for each channel.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).

    Returns:
        ndarray: Number of zero crossings per channel, shape (n_channels,).
    """
    return ant.num_zerocross(data, axis=1)

def compute_Petrosian_fd(data):
    """
    Computes the Petrosian fractal dimension for each channel.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).

    Returns:
        ndarray: Fractal dimension per channel, shape (n_channels,).
    """
    return ant.petrosian_fd(data, axis=1)


def compute_perm_entropy(data):
    """
    Computes permutation entropy for each channel.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).

    Returns:
        ndarray: Permutation entropy per channel, shape (n_channels,).
    """
    return np.array([ant.perm_entropy(each_channel, normalize=True) for each_channel in data])


def compute_detrended_fluctuation(data):
    """
    Computes detrended fluctuation analysis for each channel.

    Args:
      data (ndarray): Input data of shape (n_channels, n_times).

    Returns:
      ndarray: Detrended fluctuation per channel, shape (n_channels,).
    """
    return np.array([ant.detrended_fluctuation(each_channel) for each_channel in data])


def compute_multiscale_sample_entropy(data, sample_length=1, tolerance=None, maxscale=None):
    """
    Computes multiscale sample entropy for each channel.

    Args:
        data (ndarray): Input data of shape (n_channels, n_times).
        sample_length (int): Length of the sample.
        tolerance (float): Tolerance level for sample entropy.
        maxscale (int): Maximum scale for multiscale entropy.

    Returns:
        ndarray: Multiscale sample entropy features, shape (n_channels, maxscale).
    """
    n_channel, n_times = data.shape
    if maxscale is None:
        maxscale = n_times
    multi_en_value = np.zeros((n_channel,maxscale))
    for i_channel in range(n_channel):
        multi_en_value[i_channel, :] = np.array(
            ent.multiscale_entropy(data[i_channel, :], sample_length=sample_length, tolerance=tolerance, maxscale=maxscale))
    feature = multi_en_value.T.reshape(-1)
    return feature


def compute_multiscale_permutation_entropy(data, m=1, delay=1, scale=1):
    """
    Compute the multiscale permutation entropy for each channel in the data.

    Args:
        data (ndarray): The input data array with shape (n_channels, n_times).
        m (int): Embedding dimension for permutation entropy. Default is 1.
        delay (int): Time delay for permutation entropy. Default is 1.
        scale (int): Scale factor for multiscale permutation entropy. Default is 1.

    Returns:
        ndarray: Multiscale permutation entropy features with shape (n_channels,).
    """
    n_channel, n_times = data.shape
    multi_per_en_value = np.zeros((n_channel,scale))
    for i_channel in  range(n_channel):
        multi_per_en_value[i_channel,:]= np.array(ent.multiscale_permutation_entropy(data[i_channel,:], m, delay, scale))
    feature = multi_per_en_value.T.reshape(-1)
    return feature


def compute_fuzzy_entropy(data,m=2, tau=1, r=(.2,2), Fx='default', Logx=np.exp(1)):
    """
    Compute fuzzy entropy for each channel in the data.

    Args:
        data (ndarray): The input data array with shape (n_channels, n_times).
        m (int): Embedding dimension. Default is 2.
        tau (int): Time delay. Default is 1.
        r (float or tuple): Fuzzy function parameters. Default is (.2, 2).
        Fx (str): Fuzzy function name. Default is 'default'.
        Logx (float): Base of the logarithm for normalization. Default is e.

    Returns:
        ndarray: Fuzzy entropy features with shape (n_channels,).
    """
    n_channel, n_times = data.shape
    FuzzEn_value = np.zeros((n_channel,m))
    for i_channel in  range(n_channel):
        FuzzEn_value[i_channel],_,_=FuzzEn(data[i_channel],m=m,tau=tau,Fx=Fx,r=r,Logx=Logx)
    feature = FuzzEn_value.T.reshape(-1)
    return feature





def compute_cross_frequency_coupling(data,sfreq=250,band=np.array([[1,4], [4,8],[8,10], [10,13], [13,20], [20,30], [30,45]]),
                 mode='eeg_rhythm', low_fq_range=None, low_fq_width=2., high_fq_range='auto',
                 high_fq_width='auto', method='tort', n_surrogates=0,n_jobs=1):
    """
        Compute cross-frequency coupling using either 'eeg_rhythm' or 'Fixed_width' mode.

        Args:
            data (ndarray): The input data array with shape (n_channels, n_times).
            sfreq (int): Sampling frequency of the time signal. Default is 250 Hz.
            band (ndarray): Frequency bands for analysis with shape (fea_num, 2). Default is predefined bands.
            mode (str): Mode of computation, either 'eeg_rhythm' or 'Fixed_width'. Default is 'eeg_rhythm'.
            low_fq_range (array or list): Filtering frequencies for phase signal in 'Fixed_width' mode. Default is None.
            low_fq_width (float): Bandwidth of the band-pass filter for phase signal. Default is 2.0.
            high_fq_range (array or list or 'auto'): Filtering frequencies for amplitude signal. Default is 'auto'.
            high_fq_width (float or 'auto'): Bandwidth of the band-pass filter for amplitude signal. Default is 'auto'.
            method (str): Method for computing modulation index. Default is 'tort'.
            n_surrogates (int): Number of surrogates for z-score computation. Default is 0.
            n_jobs (int): Number of parallel jobs. Default is 1.

        Returns:
            ndarray: Cross-frequency coupling features with shape (n_channels, band_num, band_num) or (n_channels, low_fq_range.shape[0], high_fq_range.shape[0]).

        Notes:
        - This function has been abandoned.
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
        feature=feature_.T.reshape(-1)
    return feature


# def  compute_stft_2019(data,sfreq=250,win_times=10,n_fre_idx=36):
#     """
#     Compute the short-time Fourier transform (STFT) and sum the energy at fixed frequency points.
#
#     Args:
#         data (ndarray): The input data array with shape (n_channels, n_times).
#         sfreq (int): Sampling frequency of the time signal. Default is 250 Hz.
#         win_times (int): Window length in seconds. Default is 10.
#         n_fre_idx (int): Number of frequency points to sum. Default is 36.
#
#     Returns:
#         ndarray: STFT-based features with shape (n_channels, section_num, n_fre_idx).
#     """
#     from scipy.signal import stft
#     win_len = sfreq * win_times
#     n_channel, n_times = data.shape
#     section_num = n_times // win_len
#     feature = np.zeros((n_channel, section_num,n_fre_idx))
#     for i_section in range(section_num):
#         for i_channel in range(n_channel):
#             X= data[i_channel, i_section * win_len:(i_section + 1) * win_len]
#             f, t, Zxx = stft(X, fs=sfreq, window='blackman', nperseg=256, noverlap=None, nfft=512,
#                              detrend=False, boundary='zeros', padded=True)
#             Y=10*np.log(abs(np.mean(Zxx[2:n_fre_idx*2+1:2,:],axis=1)))
#             feature[i_channel, i_section] =Y
#     feature = feature.T.reshape(-1)
#     return feature


def flatten_lower_triangle(matrix):
    """
    Flatten the lower triangle of a square matrix into a 1D array.

    Args:
        matrix (ndarray): Square matrix to flatten.

    Returns:
        ndarray: Flattened array with shape (n_channel*(n_channel-1)//2,).
    """
    rows = len(matrix)
    flattened = []
    for i in range(rows):
        for j in range(i):
            flattened.append(matrix[i][j])
    flattened = np.array(flattened)
    return flattened

def reshape_to_lower_triangle(flattened_array,n_channel):
    """
    Reshape a 1D array into the lower triangle of a square matrix.

    Args:
        flattened_array (ndarray): 1D array to reshape.
        n_channel (int): Number of channels, defining the size of the square matrix.

    Returns:
        ndarray: Square matrix with shape (n_channel, n_channel).
    """
    matrix=np.zeros((n_channel,n_channel))
    count = 0
    for i in range(n_channel):
        for j in range(i):
            matrix[i][j] = flattened_array[count]
            count += 1
    return matrix




def compute_correlation_matrix(data,sfreq=250,kind="correlation",filter_bank=None,n_win=1,log = False):
    """
    Compute various types of connectivity measures from EEG data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int): Sampling frequency of the time signal. Default is 250 Hz.
        kind (str): Type of connectivity measure to compute. The available options are:
            - **Nilearn Measures**:
              - `"covariance"`: Measures the covariance between signals.
              - `"correlation"`: Measures the Pearson correlation coefficient between signals.
              - `"partial correlation"`: Measures the partial correlation, accounting for the influence of other signals.
              - `"tangent"`: Computes the tangent connectivity measure.  For the use of “tangent” see Varoquaux et al [1].
              - `"precision"`: Measures the precision of the connectivity.
           - **MNE-connectivity Measures**:
              - `"coh"`: Coherence.
              - `"cohy"`: Coherency.
              - `"imcoh"`: Imaginary part of Coherency.
              - `"cacoh"`: Canonical Coherency (CaCoh).
              - `"mic"`: Maximised Imaginary part of Coherency (MIC).
              - `"mim"`: Multivariate Interaction Measure (MIM).
              - `"plv"`: Phase-Locking Value (PLV).
              - `"ciplv"`: Corrected Imaginary PLV (ciPLV).
              - `"ppc"`: Pairwise Phase Consistency (PPC).
              - `"pli"`: Phase Lag Index (PLI).
              - `"pli2_unbiased"`: Unbiased estimator of squared PLI.
              - `"dpli"`: Directed PLI (DPLI).
              - `"wpli"`: Weighted PLI (WPLI).
              - `"wpli2_debiased"`: Debiased estimator of squared WPLI.
              - `"gc"`: State-space Granger Causality (GC).
              - `"gc_tr"`: State-space GC on time-reversed signals.
              - `"pec"`: power envolope correlation
        filter_bank (ndarray or list, optional): Band-pass filter parameters with shape (2,) [low_freq, high_freq]. Default is None (no filtering).
        n_win (int): Number of windows to split the data into. If the connectivity measure requires multiple epochs, this parameter helps in splitting one epoch into multiple parts. Default is 1.
        log (default False): If True , square and take the log before orthonalizing envelopes or computing correlations.
    Returns:
        ndarray: Flattened array of the computed connectivity matrix with shape (n_channel * n_channel,).

    Notes:
        - For certain measures like "tangent", multiple epochs are required. Ensure `n_win` is set appropriately for such measures.
        - If the `filter_bank` is specified, the data is band-pass filtered before computing the connectivity.
        - In case of an error during connectivity computation, the function returns an identity matrix and prints a warning message. Ensure the parameters are set correctly to avoid computation errors.
    References:
        [1]Gael Varoquaux, Flore Baronnet, Andreas Kleinschmidt, Pierre Fillard, and Bertrand Thirion. Detection of brain functional-connectivity difference in post-stroke patients using group-level covariance modeling. In Tianzi Jiang, Nassir Navab, Josien P. W. Pluim, and Max A. Viergever, editors, Medical image computing and computer-assisted intervention - MICCAI 2010, Lecture notes in computer science, 200–208. Berlin, Heidelberg, 2010. Springer. https://link.springer.com/chapter/10.1007/978-3-642-15705-9_25.
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
    elif kind in ['ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased', 'cohy', 'imcoh','coh','plv','gc','gc_tr','mic','mim']:

        new_data=data.reshape([n_win,n_channel,n_times//n_win])
            ###这部分计算就可以进行结果也有问题
        try:
            if filter_bank is None:
                feature_1 = spectral_connectivity_epochs(data=new_data, method=kind, mode='multitaper', sfreq=sfreq,
                                                         faverage=True, mt_adaptive=False,verbose=False)
            else:
                feature_1=spectral_connectivity_epochs(data=new_data,method=kind,mode='multitaper', sfreq=sfreq, fmin=filter_bank[0],
                                                       fmax=filter_bank[1],faverage=True, mt_adaptive=False,verbose=False)
            feature_2 = np.squeeze(feature_1.get_data("dense"))
            np.fill_diagonal(feature_2,1)
            feature = feature_2 + feature_2.T - np.diag(feature_2.diagonal())
        except:
               feature=np.eye(n_channel)
               print("feature connectivity jump")
    elif kind in ['pec']:
        new_data = data.reshape([n_win, n_channel, n_times // n_win])
        feature_1 = envelope_correlation(data=new_data,log=log,verbose =False)
        feature = np.squeeze(feature_1.get_data("dense"))

    elif kind in ['dtf','dpc']:
        feature=calculate_dtf_pdc(data,sfreq=sfreq,kind=kind,p=None,normalize_=True,filter_bank=filter_bank)

    feature = feature.T.reshape(-1)
    return feature

def compute_pac_connectivity(data,sfreq=250, method='tort', band=np.array([[4, 8],[30,45]]),
                             n_surrogates=0, mode="self", approach_pac="mean"):
    """
    Compute Phase-Amplitude Coupling (PAC) connectivity from EEG data.

    Args:
       data (ndarray): Input data with shape (n_channels, n_times).
       sfreq (int): Sampling frequency of the time signal. Default is 250 Hz.
       method (str): Method for computing PAC. Options are:
           - "tort": Tortoise method
           - "jiang": Jiang method
       band (ndarray): Frequency bands for PAC computation with shape (2, 2). Each row specifies the low and high frequencies for the band.
       n_surrogates (int): Number of surrogates for significance testing. Default is 0 (no surrogates).
       mode (str): Mode for PAC computation. Options are:
           - "self": Compute PAC for each channel with itself.
           - "non-self": Compute PAC between each pair of channels.
       approach_pac (str): Approach for summarizing PAC values. Options are:
           - "mean": Use the mean PAC value.
           - "max": Use the maximum PAC value.

    Returns:
       ndarray: Flattened array of PAC connectivity features. The shape depends on the `mode`:
           - If `mode` is "self": (n_channels,)
           - If `mode` is "non-self": (n_channels * n_channels,)

    Notes:
       - The `band` parameter specifies the frequency range for the low and high frequency bands used in PAC computation.
       - The `method` parameter determines the algorithm used for PAC calculation.
       - In "self" mode, PAC is computed for each channel individually.
       - In "non-self" mode, PAC is computed for every pair of channels.
       - The `approach_pac` parameter determines how the PAC values are aggregated: either by taking the mean or the maximum value.

    Example:
       To compute PAC using the "tort" method for each channel with itself, averaging the PAC values:
       ```python
       pac_features = compute_pac_connectivity(data, method='tort', mode='self', approach_pac='mean')
       ```
    """

    n_channel, n_times = data.shape
    low_fq_width = band[0, 1] - band[0, 0]
    high_fq_width = band[1, 1] - band[1, 0]
    low_fq_range =np.linspace(band[0,0],band[0,1],low_fq_width)
    high_fq_range = np.linspace(band[1, 0], band[1, 1], high_fq_width)
    c = Comodulogram(fs=sfreq, low_fq_range=low_fq_range, high_fq_range=high_fq_range,method=method,
                     n_surrogates=n_surrogates,n_jobs=-1)
    if mode== "self":
        feature=np.zeros((n_channel))
        for N_channel in range(n_channel):
            sig = data[N_channel, :]
            pac_matrix=c.fit(sig).comod_
            if approach_pac=="mean":
                  feature[N_channel] = np.mean(pac_matrix)
            elif approach_pac=="max":
                feature[N_channel] = np.max(pac_matrix)

    elif mode== "non-self":
        feature = np.zeros((n_channel,n_channel))
        for i_channel in range(n_channel):
            for j_channel in range(n_channel):
                sig_low = data[i_channel, :]
                sig_high = data[j_channel, :]
                pac_matrix = c.fit(sig_low,sig_high).comod_
                if approach_pac == "mean":
                    feature[i_channel,j_channel] =  np.mean(pac_matrix)
                elif approach_pac == "max":
                    feature[i_channel, j_channel] = np.max(pac_matrix)
    feature = feature.reshape(-1)
    return feature

def compute_pac_connectivity_mod(data,sfreq=250, method='tort', band=np.array([[4, 8],[30,45]]),
                             n_surrogates=0, mode="self", approach_pac="mean"):
    """
    Compute Phase-Amplitude Coupling (PAC) connectivity from EEG data.

    Args:
       data (ndarray): Input data with shape (n_channels, n_times).
       sfreq (int): Sampling frequency of the time signal. Default is 250 Hz.
       method (str): Method for computing PAC. Options are:
           - "tort": Tortoise method
           - "jiang": Jiang method
       band (ndarray): Frequency bands for PAC computation with shape (2, 2). Each row specifies the low and high frequencies for the band.
       n_surrogates (int): Number of surrogates for significance testing. Default is 0 (no surrogates).
       mode (str): Mode for PAC computation. Options are:
           - "self": Compute PAC for each channel with itself.
           - "non-self": Compute PAC between each pair of channels.
       approach_pac (str): Approach for summarizing PAC values. Options are:
           - "mean": Use the mean PAC value.
           - "max": Use the maximum PAC value.

    Returns:
       ndarray: Flattened array of PAC connectivity features. The shape depends on the `mode`:
           - If `mode` is "self": (n_channels,)
           - If `mode` is "non-self": (n_channels * n_channels,)

    Notes:
       - The `band` parameter specifies the frequency range for the low and high frequency bands used in PAC computation.
       - The `method` parameter determines the algorithm used for PAC calculation.
       - In "self" mode, PAC is computed for each channel individually.
       - In "non-self" mode, PAC is computed for every pair of channels.
       - The `approach_pac` parameter determines how the PAC values are aggregated: either by taking the mean or the maximum value.

    Example:
       To compute PAC using the "tort" method for each channel with itself, averaging the PAC values:
       ```python
       pac_features = compute_pac_connectivity(data, method='tort', mode='self', approach_pac='mean')
       ```
    """

    n_channel, n_times = data.shape
    low_fq_width = band[0, 1] - band[0, 0]
    high_fq_width = band[1, 1] - band[1, 0]
    low_fq_range =np.linspace(band[0,0],band[0,1],low_fq_width+1)
    high_fq_range = np.linspace(band[1, 0], band[1, 1], high_fq_width+1)
    c = Comodulogram(fs=sfreq, low_fq_range=low_fq_range, high_fq_range=high_fq_range,method=method,
                     n_surrogates=n_surrogates,n_jobs=-1)
    if mode== "self":
        feature=np.zeros((n_channel))
        for N_channel in range(n_channel):
            sig = data[N_channel, :]
            pac_matrix=c.fit(sig).comod_
            if approach_pac=="mean":
                  feature[N_channel] = np.mean(pac_matrix)
            elif approach_pac=="max":
                feature[N_channel] = np.max(pac_matrix)

    elif mode== "non-self":
        feature = np.zeros((n_channel,n_channel))
        for i_channel in range(n_channel):
            for j_channel in range(n_channel):
                sig_low = data[i_channel, :]
                sig_high = data[j_channel, :]
                pac_matrix = c.fit(sig_low,sig_high).comod_
                if approach_pac == "mean":
                    feature[i_channel,j_channel] =  np.mean(pac_matrix)
                elif approach_pac == "max":
                    feature[i_channel, j_channel] = np.max(pac_matrix)
    feature = feature.reshape(-1)
    return feature

def compute_correlation_dimension(data,emb_dim=10):
    """
    Args:
        data:        ndarray,           shape (n_channels, n_times)
        emb_dim:     int                嵌入维度默认为10
    Returns:         feature            shape (n_channels)
    """
    import nolds
    n_channel, n_times = data.shape
    feature = np.zeros((n_channel))
    for i_channel in  range(n_channel):
        feature[i_channel] = nolds.corr_dim(data[i_channel,:],emb_dim)
    return feature
def compute_dispersion_entropy(data,classes=10,scale=1,emb_dim=2,delay=1,
                                   mapping_type='cdf',de_normalize=False, A=100,Mu=100,return_all=False,warns=True):
    """
    Args:
        data:        ndarray,           shape (n_channels, n_times)
        classes: number of classes - (levels of quantization of amplitude) (default=10)
        emb_dim: embedding dimension,
        delay  : time delay (default=1)
        scale  : downsampled signal with low resolution  (default=1)  - for multipscale dispersion entropy
        mapping_type: mapping method to discretizing signal (default='cdf')
               : options = {'cdf','a-law','mu-law','fd'}
        A  : factor for A-Law- if mapping_type = 'a-law'
        Mu : factor for μ-Law- if mapping_type = 'mu-law'

        de_normalize: (bool) if to normalize the entropy, to make it comparable with different signal with different
                     number of classes and embeding dimensions. default=0 (False) - no normalizations

        if de_normalize=1:
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
    """
    import spkit
    n_channel, n_times = data.shape
    feature = np.zeros((n_channel))
    for i_channel in  range(n_channel):
          feature[i_channel],_ = spkit.dispersion_entropy(data[i_channel,:],classes,scale,emb_dim,delay,
                                   mapping_type,de_normalize, A,Mu,return_all,warns)
    return feature
def compute_aperiodic_periodic_offset_exponent_cf(data, sfreq=250, n=1024,freq_range=None, method="welch"):
    """
    Compute aperiodic and periodic parameters of the power spectrum from EEG data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int): Sampling frequency of the signal. Default is 250 Hz.
        n (int): Number of frequency points for the Fourier transform. Typically set to the number of data points.
        freq_range (list or None): Frequency range for fitting with FOOOF. Default is None.
        method (str): Method for computing the power spectrum. Options are:
            - "fft": Fast Fourier Transform
            - "welch": Welch's method

    Returns:
        ndarray: Flattened array of aperiodic and periodic parameters with shape (n_channels * 2,).

    Notes:
        - Computes the offset and exponent of the aperiodic component and the periodic component of the power spectrum.
        - Uses the FOOOFGroup for fitting the power spectrum.
    """

    n_channel, n_times = data.shape
    feature = np.zeros((n_channel,2))
    if method =="fft":
        freqs = np.fft.fftfreq(n_times, 1 / sfreq)
        spectrum_frequencies = np.abs(np.fft.fft(data,n=n_times))
    elif method =="welch":
        freqs, spectrum_frequencies = welch(data, sfreq, nperseg=sfreq)
    spectrum_frequencies[:, freqs >= 0]
    fg1 = FOOOFGroup(verbose = False)
    fg1.fit(freqs, np.square(spectrum_frequencies)/(n**2), freq_range=freq_range)
    #[Offset, Exponent].
    feature[:,:2] = fg1.get_params("aperiodic_params")
    #[CF, PW, BW]..
    # peak_para=fg1.get_params('peak_params')
    # for i_channel in range(n_channel):
    #     idx_i_channel = np.where(peak_para[:,3] == i_channel)[0]
    #     peak_i_channel = peak_para[idx_i_channel,:]
    #     feature[i_channel,2] = peak_i_channel[np.argmax(peak_i_channel[:,1]),0]
    # bands = Bands({'alpha': [8, 12]})
    # feature[:,3] = get_band_peak_fg(fg1, bands.alpha)[:,0]
    feature = feature.T.reshape(-1)
    return feature

def compute_offset_exponent_cf(data,sfreq=250,n=1024):
    """
    Compute the offset and exponent of the power spectrum from EEG data.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int): Sampling frequency of the signal. Default is 250 Hz.
        n (int): Number of frequency points for the Fourier transform. Typically set to the number of data points.

    Returns:
        ndarray: Flattened array of offset and exponent parameters with shape (n_channels * 2,).

    Notes:
        - Computes the median frequency and the spectral slope (offset and exponent) using `compute_spect_slope`.
        - The spectral slope is inverted in the returned feature array.
    """
    from mne_features.univariate import compute_spect_slope
    n_channel, n_times = data.shape
    feature = np.zeros((n_channel,2))
    feature[:,2]=compute_Median_Frequency(data,sfreq=250,band=np.array([[8,12]]),N=n)
    # print(feature[:,2])
    slope_para=compute_spect_slope(sfreq, data, fmin=0.1, fmax=50,with_intercept=True, psd_method='welch', psd_params=None)
    slope_para=slope_para.reshape(n_channel,4)
    # print(slope_para)
    # print(slope_para.shape)
    # intercept, slope,
    feature[:, :2]=slope_para[:, :2]
    feature[:,1]=-feature[:,1]
    feature = feature.T.reshape(-1)
    return feature

# def compute_relative_power(data, sfreq=100, freq_bands=np.array([0.5, 4]), total_band = np.array([0.5, 50]),
#                            ratios=None, ratios_triu=False,psd_method='welch', log=False, psd_params=None):
#     """
#     Args:
#         data:        ndarray,           shape (n_channels, n_times)
#         sfreq:       sfreq              信号采样频率
#         freq_bands:  narray             定义方式类似于compute_pow_freq_bands
#         total_band:  narray             np.array([0.5, 50]
#         normalize:
#         ratios:
#         ratios_triu:
#         psd_method:
#         log:
#         psd_params:
#
#     Returns:
#
#     """
#     band_power = compute_pow_freq_bands(sfreq, data, freq_bands=freq_bands,
#                            normalize=False, ratios=ratios, ratios_triu=ratios_triu,
#                            psd_method=psd_method, log=log, psd_params=psd_params)
#     total_power= compute_pow_freq_bands(sfreq, data, freq_bands=total_band,
#                            normalize=False, ratios=ratios, ratios_triu=ratios_triu,
#                            psd_method=psd_method, log=log, psd_params=psd_params)
#     n_band = int(band_power.shape[0]/total_power.shape[0])
#     total_power = np.tile(total_power, (n_band, 1))
#     relaPower = (band_power/total_power).T.reshape(-1)
#     return relaPower

def get_power_from_channel(data,wind,windowsover,i_channel,channel,sfreq,freq1,freq2):
    """
    Compute the power of a specified frequency band for a given channel.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        wind (int): Window length for the spectrogram.
        windowsover (int): Number of overlapping windows for the spectrogram.
        i_channel (int): Index of the channel to analyze.
        channel (list): List of channel names.
        sfreq (int): Sampling frequency of the signal.
        freq1 (float): Lower bound of the frequency range.
        freq2 (float): Upper bound of the frequency range.

    Returns:
        ndarray: Power of the specified frequency band for the given channel.

    Notes:
        - Uses `plt.specgram` to compute the power spectrum of the channel data.
    """
    L = channel.index(i_channel)
    signL = data[L, :]
    powerL, freqsL, t, _ = plt.specgram(signL, mode='psd', NFFT=wind, noverlap=windowsover, Fs=sfreq)
    indFreqs = [i for i, x in enumerate(freqsL) if freq1 <= x <= freq2]
    pow_L = powerL[indFreqs, 0]
    return pow_L

def compute_alpha_asymetry(data, sfreq=100, freq1=8, freq2=12, left='F3', right='F4', channel=None, mode="eeglab"):
    """
    Compute the alpha asymmetry between two specified channels.

    Args:
        data (ndarray): Input data with shape (n_channels, n_times).
        sfreq (int): Sampling frequency of the signal. Default is 100 Hz.
        freq1 (float): Lower bound of the alpha frequency range.
        freq2 (float): Upper bound of the alpha frequency range.
        left (str): Name of the left channel.
        right (str): Name of the right channel.
        channel (list or None): List of channel names. Default is None, in which case a default list is used.
        mode (str): Method for computing alpha asymmetry. Options are:
            - "eeglab": Method used in EEGLAB.
            - "definition_ln": Logarithmic difference.
            - "definition_ratio": Ratio difference.
            - "definition_lnratio": Logarithmic ratio difference.
            - "definition_ln_rel": Logarithmic relative difference.
            - "definition_ratio_rel": Ratio relative difference.
            - "definition_lnratio_rel": Logarithmic ratio relative difference.

    Returns:
        ndarray: Array of alpha asymmetry values with shape (n_channels,).

    Notes:
        - Computes alpha asymmetry using different methods depending on the `mode` parameter.
        - If `mode` is "eeglab", uses the power spectral density (PSD) of the specified channels.
    """
    if channel is None:
        channel = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4',
                   'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3',
                   'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    n_channel, length = data.shape
    try:
        if mode =="eeglab":
                wind = int(np.floor(length / 1.5))
                windowsover = int(np.floor(length / 1.5 / 2))
                pow_L = get_power_from_channel(data,wind,windowsover,left,channel,sfreq,freq1,freq2)
                pow_R = get_power_from_channel(data, wind, windowsover, right, channel,sfreq,freq1,freq2)
                FAA = np.mean(np.abs(np.log(pow_R) - np.log(pow_L)))
        elif "definition" in mode:
            info = mne.create_info(channel, sfreq)
            raw = mne.io.RawArray(data, info)
            psd = raw.compute_psd(method="welch", fmin=0.5, fmax=sfreq / 2, picks="all")
            alpha_mask = ((psd.freqs >= freq1) & (psd.freqs <= freq2))

            F3 = psd.copy().pick([left])
            F4 = psd.copy().pick([right])
            F3_total = F3.get_data([left]).squeeze()
            F4_total = F4.get_data([right]).squeeze()

            F3_alpha = F3_total[alpha_mask]
            F4_alpha = F4_total[alpha_mask]

            F3_rel_alpha = F3_alpha.mean() / F3_total.mean()
            F4_rel_alpha = F4_alpha.mean() / F4_total.mean()
            if mode == "definition_ln":
                FAA = np.mean(np.abs(np.log(F4_alpha) - np.log(F3_alpha)))
            elif mode == "definition_ratio":
                FAA = np.mean(np.abs((F4_alpha - F3_alpha) / (F3_alpha + F4_alpha)))
            elif mode == "definition_lnratio":
                FAA = np.mean(np.abs((np.log(F4_alpha) - np.log(F3_alpha)) / (np.log(F3_alpha) + np.log(F4_alpha))))
            elif mode == "definition_ln_rel":
                FAA = np.abs(np.log(F4_rel_alpha) - np.log(F3_rel_alpha))
            elif mode == "definition_ratio_rel":
                FAA = np.abs((F4_rel_alpha - F3_rel_alpha) / (F3_rel_alpha + F4_rel_alpha))
            elif mode == "definition_lnratio_rel":
                FAA = np.abs(
                    (np.log(F4_rel_alpha) - np.log(F3_rel_alpha)) / (np.log(F3_rel_alpha) + np.log(F4_rel_alpha)))
        feature = np.zeros(n_channel) + FAA
    except:
        feature = np.zeros(n_channel)
    return feature
def compute_pow_freq_bands_remove_aperiodic(data, sfreq=250, freq_bands=np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 40]]),
                                           psd_method='welch', log=False, freq_range=None):
    """
    Compute the power in specified frequency bands after removing aperiodic components.

    Args:
        data (ndarray): Shape (n_channels, n_times). The input signal data.
        sfreq (int): The sampling frequency of the signal.
        freq_bands (ndarray): An array of frequency bands for power computation. Shape (n_bands, 2).
        psd_method (str): The method to use for computing the power spectral density ('fft' or 'welch').
        log (bool): Whether to apply a logarithm to the resulting power values.
        freq_range (tuple or None): The frequency range for aperiodic component fitting. If None, no range is used.

    Returns:
        ndarray: The power in each frequency band after removing aperiodic components. Flattened array of shape (n_channels * n_bands,).
    """
    n_channel, n_times = data.shape
    if psd_method == "fft":
        freqs = np.fft.fftfreq(n_times, 1/sfreq)
        spectrum_frequencies = np.abs(np.fft.fft(data, n=n_times))
    elif psd_method == "welch":
        freqs, spectrum_frequencies = welch(data, sfreq, nperseg=sfreq)
    spectrum_frequencies[:,freqs >= 0]
    fg1 = FOOOFGroup(verbose=False)
    fg1.fit(freqs, np.square(spectrum_frequencies)/(sfreq*n_times), freq_range=freq_range)
    offset_ex_list = fg1.get_params('aperiodic_params')
    aperiodic_value = np.zeros_like(spectrum_frequencies)
    for i_channel in range(n_channel):
        for idx in np.where(freqs > 0)[0]:
            aperiodic_value[i_channel, idx] = np.power(10, offset_ex_list[i_channel, 0] -
                                       offset_ex_list[i_channel, 1] * np.log10(freqs[idx]))
    all_psd = pow_freq_bands_from_spectrum(freq_bands, freqs, spectrum_frequencies)
    ap_psd =  pow_freq_bands_from_spectrum(freq_bands, freqs, aperiodic_value)
    remaind_psd = all_psd-ap_psd
    if log:
        remaind_psd = np.log10(remaind_psd)
    return remaind_psd.T.reshape(-1)
def compute_pow_freq_bands_cd(data, sfreq=250, freq_bands=np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 40]]),
                                           psd_method='welch', log=False):
    """
    Compute the power in specified frequency bands from the power spectral density.

    Args:
        data (ndarray): Shape (n_channels, n_times). The input signal data.
        sfreq (int): The sampling frequency of the signal.
        freq_bands (ndarray): An array of frequency bands for power computation. Shape (n_bands, 2).
        psd_method (str): The method to use for computing the power spectral density ('fft' or 'welch').
        log (bool): Whether to apply a logarithm to the resulting power values.

    Returns:
        ndarray: The power in each frequency band. Flattened array of shape (n_channels * n_bands,).
    """
    n_channel, n_times = data.shape
    if psd_method == "fft":
        freqs = np.fft.fftfreq(n_times, 1/sfreq)
        spectrum_frequencies = np.abs(np.fft.fft(data, n=n_times))
    elif psd_method == "welch":
        freqs, spectrum_frequencies = welch(data, sfreq, nperseg=sfreq)
    spectrum_frequencies[:,freqs >= 0]
    all_psd = pow_freq_bands_from_spectrum(freq_bands, freqs, spectrum_frequencies)
    if log:
        all_psd = np.log10(all_psd)
    return all_psd.T.reshape(-1)

def pow_freq_bands_from_spectrum(freq_bands,freqs,spectrum_frequencies):
    """
    Compute the power in specified frequency bands from the power spectral density.

    Args:
        freq_bands (ndarray): An array of frequency bands for power computation. Shape (n_bands, 2).
        freqs (ndarray): Array of frequency values corresponding to the power spectrum.
        spectrum_frequencies (ndarray): The power spectrum of the data. Shape (n_channels, n_freqs).

    Returns:
        ndarray: The power in each frequency band. Shape (n_channels, n_bands).
    """
    n_bands = freq_bands.shape[0]
    n_channels = spectrum_frequencies.shape[0]
    feature = np.zeros((n_channels, n_bands))
    for index, i_band in enumerate(freq_bands):
        feature[:,index] = np.sum(spectrum_frequencies[:,np.where((freqs >= i_band[0]) & (freqs <= i_band[1]))[0]], axis=1)
    return feature



def compute_periodic_pac_connectivity(data, sfreq=250, n=1024, method="tort",
                                      band=np.array([[4, 8],[30,45]]), n_surrogates=0, mode="self", approach_pac="mean"):
    """
    Compute periodic phase-amplitude coupling (PAC) connectivity from the signal data.

    Args:
        data (ndarray): Shape (n_channels, n_times). The input signal data.
        sfreq (int): The sampling frequency of the signal.
        n (int): The number of points for Fourier transform.
        method (str): The method to use for PAC computation ('tort' or others).
        band (ndarray): An array specifying the frequency bands for PAC computation. Shape (2, 2).
        n_surrogates (int): The number of surrogate data to compute for significance testing.
        mode (str): The mode of PAC computation ('self' or others).
        approach_pac (str): The approach to compute PAC ('mean' or others).

    Returns:
        ndarray: The PAC connectivity feature. Flattened array of shape (n_channels * n_channels,).
    """

    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    n_channel, n_times = data.shape
    n=n_times
    spectrum_frequencies = np.zeros((n_channel,n))
    spectrum_phase = np.zeros((n_channel,n))
    reconstructed_signal = np.zeros_like(data)
    freqs = np.fft.fftfreq(n, 1 / sfreq)
    for i_channel in range(n_channel):
        fft_result = np.fft.fft(data[i_channel,:],n=n)
        spectrum_frequencies[i_channel, :]=np.abs(fft_result)
        spectrum_phase[i_channel,:] = np.angle(fft_result)
    fg1 = FOOOFGroup(verbose=False)
    fg1.fit(freqs, spectrum_frequencies)

    F=freqs
    gaussian_values = np.zeros_like(spectrum_frequencies)
    aps1 = fg1.get_params('gaussian_params')
    #print(aps1)
    gaussian_list=aps1[:,3]
    for i_channel in range(n_channel):
         i_channel_dx=np.where(gaussian_list==i_channel)[0]
         for idx in i_channel_dx:
            gaussian_values[i_channel,:] = aps1[idx,1] * norm.pdf(F, aps1[idx,0], aps1[idx,2])+gaussian_values[i_channel,:]
    gaussian_values = np.power(10, gaussian_values)
    for idx in np.where(freqs < 0)[0]:
        fre__ = -freqs[idx]
        try:
            gaussian_values[:, idx] = gaussian_values[:, np.where(freqs == fre__)[0][0]]
        except:
            print(freqs[idx])


    for i_channel in range(n_channel):
        # amplitude_interp = interp1d(freqs, gaussian_values[i_channel,:], kind='linear', fill_value='extrapolate')
        # phase_interp = interp1d(freqs, spectrum_phase[i_channel,:], kind='linear', fill_value='extrapolate')
        # new_sampling_rate = n_times / (gaussian_values[i_channel,:].shape[0] - 1) * sfreq
        # new_freq_vector = np.fft.fftfreq(n_times, d=1 / new_sampling_rate)
        #
        # # 插值得到新的幅值信号和相位信号
        # new_amplitude = amplitude_interp(new_freq_vector)
        # new_phase = phase_interp(new_freq_vector)
        # reconstructed_signal[i_channel, :] = np.fft.ifft(
        #     new_amplitude * np.exp(1j * new_phase))

        reconstructed_signal[i_channel,:] = np.fft.ifft(gaussian_values[i_channel,:] * np.exp(1j * spectrum_phase[i_channel,:]))

    # plt.plot(reconstructed_signal[i_channel,:])
    # plt.xlabel('time')
    # plt.ylabel('reconstructed_signal')
    # plt.show()


    feature=compute_pac_connectivity(reconstructed_signal,sfreq=sfreq, method=method, band=band, n_surrogates=n_surrogates,mode=mode,approach_pac=approach_pac)
    return feature
    
   
    
    







