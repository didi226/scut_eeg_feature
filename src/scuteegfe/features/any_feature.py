import numpy as np
from scipy.signal import hilbert
from scipy.stats import iqr
from PyEMD import EMD
import antropy as ant
from pyentrp import entropy as ent
from ..HOSA.conventional.bicoherence import bicoherence


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
    :return: ndarray, shape (n_channels, n_length*EMD_params*EMD_length)
    """
    EMD_length = sfreq * EMD_times
    n_channel, n_times = data.shape
    n_length = n_times // EMD_length
    signal_imfs = np.zeros((n_channel, n_length, EMD_params, EMD_length))
    emd = EMD()
    for N_length in range(n_length):
        for N_channel in range(n_channel):
            IMFs = emd.emd(data[N_channel, N_length * EMD_length:(N_length + 1) * EMD_length])
            signal_imfs[N_channel, N_length, :, :] = IMFs[0:EMD_params, :]
    feature = signal_imfs.reshape(-1)
    return feature


def compute_test2(data):
    return np.mean(data, axis=-1)


def compute_IQR(data):
    """
    四分位数
    :param data: ndarray, shape (n_channels, n_times)
    :return: ndarray, shape (n_channels,)
    """
    return iqr(data, axis=1)


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
