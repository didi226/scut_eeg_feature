## Description

`scuteegfe`[mne-feature](https://mne.tools/mne-features/api.html)  is an encapsulation that adds more features on top of it, especially the feature of functional connectivity.

You can get more information and usage about the package from the  [Document ](https://scut-eeg-feature.readthedocs.io/en/latest/).

## Installation

```bash
pip install scuteegfe
```
# Usage
Usage reference[mne-feature](https://mne.tools/mne-features/api.html)，The usage method should be consistent with it.
```python
from scuteegfe import Feature
import numpy as np

n_samples = 100
n_channels = 32
n_timesamples = 250
data = np.random.randint(0,100,size=(n_samples, n_channels, n_timesamples))

Fea = Feature(data,selected_funcs=Feature.mne_defined_funcs, funcs_params=None, n_jobs=8)
features = Fea.features
feature_names = Fea.feature_names
```

If you think this toolkit or the results are helpful to you and your research, please cite us!

```
@article{chen17eeg,
  title={An EEG-Based Attention Recognition Method: Fusion of Time Domain, Frequency Domain and Nonlinear Dynamics Features},
  author={Chen, Di and Huang, Haiyun and Pan, Jiahui and Bao, Xiaoyu and Li, Yuanqing},
  journal={Frontiers in Neuroscience},
  volume={17},
  pages={1194554},
  publisher={Frontiers}
}
```

