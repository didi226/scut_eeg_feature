# 简介
`scuteegfe`是[mne-feature](https://mne.tools/mne-features/api.html)的一个封装，并在其基础之上新增了更多的特征。
# 构建
```bash
python -m build
```

# 安装
```bash
pip install --editable .
```
注：`--editable`表示以开发模式安装，对源码的改动不用重新构建和安装也能生效。

# 使用
用法参考[mne-feature](https://mne.tools/mne-features/api.html)，与其保持一致。
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

