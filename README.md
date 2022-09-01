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