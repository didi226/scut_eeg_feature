import unittest
import numpy as np

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


if __name__ == '__main__':
    unittest.main()
