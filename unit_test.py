import unittest
import numpy as np
import matplotlib.pyplot as plt

from scuteegfe import Feature


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


if __name__ == '__main__':
    unittest.main()
