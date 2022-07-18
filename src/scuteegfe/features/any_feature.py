import numpy as np
from scipy.signal import hilbert

def compute_Hilbert_abs(data):
    n_channel,n_times=data.shape
    feature=abs(hilbert(data)).reshape(-1)
    #feature=feature[np.newaxis, :];
    return feature


def compute_test2(data):
    return np.mean(data, axis=-1)
