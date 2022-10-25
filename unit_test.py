import unittest
import numpy as np
import matplotlib.pyplot as plt
from scuteegfe import Feature

from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf


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
 #   self.assertTrue(data.ndim == 3)
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

        fea1 = Feature(data, selected_funcs=Feature.funcs_subset_no_spect_slope)
        fea2 = Feature(data2, selected_funcs=Feature.funcs_subset_no_spect_slope)

        Feature.plot_feature_sns(fea1, fea2, ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        plt.show()
        _, p = Feature.ttest_feature(fea1, fea2, ['ch1', 'ch2', 'ch3', 'ch4', 'ch5'])
        plt.show()
        self.assertTrue(True)

    def test_get_data(self):
        data, _ = get_data_example_motor_image()
        fea = Feature(data, selected_funcs=['mean','std'])
        n_sample_list=np.arange(3,7)
        fea_part=fea.get_data(n_sample_list=n_sample_list)
        print(fea_part.shape)
        self.assertTrue((fea_part == fea.features[n_sample_list]).all())
        fea_part_0=fea.get_data(n_sample_list=1)
        self.assertTrue(fea_part_0.ndim==fea.features.ndim)


    def test_get_item(self):
        data,_=get_data_example_motor_image()
        #(45, 64, 801)
        fea = Feature(data, selected_funcs=['mean','std'])
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
        # test non int
        #fea_new_2 = fea[4.3]

if __name__ == '__main__':
    unittest.main()

