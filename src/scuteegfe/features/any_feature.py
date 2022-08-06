import numpy as np
from scipy.signal import hilbert
from PyEMD import EMD
from ..HOSA.conventional.bicoherence import bicoherence

def compute_Hilbert_abs(data):
    n_channel,n_times=data.shape
    feature=abs(hilbert(data)).reshape(-1)
    #feature=feature[np.newaxis, :];
    return feature
def compute_EMD(data,sfreq=250,EMD_times=1,EMD_params=6):
    EMD_length=sfreq*EMD_times
    n_channel, n_times = data.shape
    n_length = n_times // EMD_length
    signal_imfs = np.zeros((n_channel, n_length, EMD_params, EMD_length));
    emd = EMD()
    for N_length in range(n_length):
        for N_channel in range(n_channel):
          IMFs = emd.emd(data[N_channel,N_length*EMD_length:(N_length+1)*EMD_length])
          signal_imfs[N_channel,N_length,:,:]=IMFs[0:EMD_params,:]
    feature=signal_imfs.reshape(-1)
    return feature
def compute_hosa_(data):

    return



def compute_test2(data):
    return np.mean(data, axis=-1)
