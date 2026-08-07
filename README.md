# scuteegfe

[![PyPI version](https://img.shields.io/pypi/v/scuteegfe.svg)](https://pypi.org/project/scuteegfe/)
[![Python versions](https://img.shields.io/pypi/pyversions/scuteegfe.svg)](https://pypi.org/project/scuteegfe/)
[![Documentation Status](https://readthedocs.org/projects/scut-eeg-feature/badge/?version=latest)](https://scut-eeg-feature.readthedocs.io/en/latest/)
[![GitHub stars](https://img.shields.io/github/stars/didi226/scut_eeg_feature?style=flat)](https://github.com/didi226/scut_eeg_feature)

**EEG and time-series feature extraction with a familiar MNE-Features-style interface.**

`scuteegfe` extends [MNE-Features](https://mne.tools/mne-features/) with additional entropy, nonlinear dynamics, spectral, aperiodic, functional-connectivity, and cross-frequency-coupling features. It provides one interface for extracting channel-wise features and connectivity matrices from epoched EEG or other multichannel time-series data.

- **Documentation:** https://scut-eeg-feature.readthedocs.io/en/latest/
- **PyPI:** https://pypi.org/project/scuteegfe/
- **Source code:** https://github.com/didi226/scut_eeg_feature
- **Issues:** https://github.com/didi226/scut_eeg_feature/issues

## Highlights

- Uses an interface consistent with `mne-features`.
- Accepts epoched data with shape `(n_epochs, n_channels, n_times)`.
- Includes MNE-Features functions and more than 30 additional feature functions.
- Covers entropy, complexity, fractal, spectral, aperiodic, connectivity, PAC, AAC, EMD, ARMA, and distance-based features.
- Supports function-specific parameters through the `"<feature>__<parameter>"` convention.
- Supports parallel feature extraction through `n_jobs`.
- Preserves matrix outputs for connectivity features.

## Installation

`scuteegfe` requires Python 3.10 or later.

```bash
python -m pip install -U scuteegfe
```

For development:

```bash
git clone https://github.com/didi226/scut_eeg_feature.git
cd scut_eeg_feature
python -m pip install -e .
```

## Quick start

```python
import numpy as np
from scuteegfe import Feature

sfreq = 250
rng = np.random.default_rng(42)

# Shape: (n_epochs, n_channels, n_times)
data = rng.standard_normal((10, 8, sfreq * 4))

extractor = Feature(data, sfreq=sfreq, selected_funcs=["pow_freq_bands", "Shannon_entropy", "Petrosian_fd"],
    funcs_params={"pow_freq_bands__freq_bands": np.array([[1, 4], [4, 8],[8, 13],[13, 30],[30, 50],]),
        "pow_freq_bands__normalize": False,
        "Shannon_entropy__sfreq": sfreq,
        "Shannon_entropy__win_times": 1,
    },
)

features = extractor.features
feature_names = extractor.feature_names

print(features.shape)
print(feature_names)
```

Most channel-wise feature sets are returned with shape:

```text
(n_epochs, n_channels, n_features)
```

Some matrix-valued functions, such as `correlation_matrix`, return:

```text
(n_epochs, n_channels, n_channels)
```

## Selecting features

Feature names are passed to `selected_funcs`.

For example,

```python
selected_funcs=["fuzzy_entropy"]
```

Function-specific parameters use two underscores:

```python
funcs_params={
    "fuzzy_entropy__m": 3,
    "fuzzy_entropy__tau": 1,
}
```

For functions that use the sampling frequency, pass `sfreq` to both `Feature` and the corresponding function parameters when required:

```python
extractor = Feature(data, sfreq=sfreq,
    selected_funcs=["DFA", "Renyi_Entropy", "Tsallis_Entropy"],
    funcs_params={"DFA__sfreq": sfreq,"Renyi_Entropy__sfreq": sfreq,"Tsallis_Entropy__sfreq": sfreq,},)
```

## Using MNE-Features functions

MNE-Features functions can be selected through the same interface:

```python
extractor = Feature(
    data,
    sfreq=sfreq,
    selected_funcs=["mean", "variance", "pow_freq_bands"],
    funcs_params={"pow_freq_bands__freq_bands": np.array([[1, 4],[4, 8],[8, 13],[13, 30],[30, 50],]),
    "pow_freq_bands__normalize": False,
    },
)
```
See the [MNE-Features API](https://mne.tools/mne-features/api.html) for the available MNE-defined functions and their parameters.

## Functional connectivity

`correlation_matrix` provides a unified interface for covariance-, correlation-, spectral-, phase-, envelope-, and directed-connectivity measures.

```python
connectivity = Feature(data,sfreq=sfreq,
    selected_funcs=["correlation_matrix"],
    funcs_params={
        "correlation_matrix__sfreq": sfreq,
        "correlation_matrix__kind": "pec",
        "correlation_matrix__filter_bank": [8, 13],
        "correlation_matrix__n_win": 4,
    },
)
connectivity_matrices = connectivity.features
print(connectivity_matrices.shape)
```

The result has shape `(n_epochs, n_channels, n_channels)`.

Common `kind` options include covariance, correlation, partial correlation, tangent, precision, coherence, coherency, imaginary coherence, PLV, PPC, PLI, weighted PLI, Granger causality, and power-envelope correlation. See the [connectivity example](https://scut-eeg-feature.readthedocs.io/en/latest/example/connectivity.html) for details.

## Phase-amplitude coupling

PAC can be calculated within each channel or between channel pairs.

```python
pac = Feature(data,sfreq=sfreq, selected_funcs=["pac_connectivity"],
    funcs_params={"pac_connectivity__sfreq": sfreq,
        "pac_connectivity__band": np.array([[4, 8],[30, 45],]),
        "pac_connectivity__method": "tort",
        "pac_connectivity__mode": "self",
        "pac_connectivity__approach_pac": "mean",
    },
)
pac_features = pac.features
```


## Available custom features

The names below are the values used in `selected_funcs`.

### Entropy, complexity, and nonlinear dynamics

| Feature | Description |
|---|---|
| `DFA` | Windowed detrended fluctuation analysis for long-range correlations. |
| `Shannon_entropy` | Windowed Shannon entropy. |
| `Renyi_Entropy` | Windowed Rényi entropy. |
| `Tsallis_Entropy` | Windowed Tsallis entropy. |
| `wavelet_entropy` | Wavelet-band energy or entropy features. |
| `Petrosian_fd` | Petrosian fractal dimension. |
| `perm_entropy` | Normalized permutation entropy. |
| `detrended_fluctuation` | Channel-wise detrended fluctuation value. |
| `multiscale_sample_entropy` | Sample entropy across multiple scales. |
| `multiscale_permutation_entropy` | Permutation entropy across multiple scales. |
| `fuzzy_entropy` | Fuzzy entropy with configurable embedding parameters. |
| `correlation_dimension` | Correlation dimension with configurable embedding dimension. |
| `dispersion_entropy` | Dispersion entropy with configurable mapping and normalization. |

### Spectral and aperiodic features

| Feature | Description |
|---|---|
| `Harmonic_Parameters` | FFT amplitude at the center of predefined frequency bands. |
| `Median_Frequency` | Median frequency within specified frequency bands. |
| `alpha_asymetry` | Alpha-band asymmetry between two selected channels. The spelling is retained for API compatibility. |
| `pow_freq_bands_cd` | Power within user-defined frequency bands. |
| `pow_freq_bands_remove_aperiodic` | Band power after removing the fitted aperiodic component. |
| `aperiodic_periodic_offset_exponent_cf` | FOOOF-based aperiodic offset and exponent. |
| `offset_exponent_cf` | Spectral intercept and slope calculated over a selected frequency range. |

### Connectivity and cross-frequency coupling

| Feature | Description | Status |
|---|---|---|
| `Coherence` | Band-wise coherence between selected channel pairs. | Available |
| `correlation_matrix` | Multichannel functional-connectivity matrix. | Available |
| `pac_connectivity` | Phase-amplitude coupling within or between channels. | Available |
| `aac_connectivity` | Amplitude-amplitude coupling using Morlet or multitaper TFR. | Available |
| `pac_connectivity_mod` | Alternative PAC implementation. | Available |
| `periodic_pac_connectivity` | PAC calculated after isolating periodic spectral activity. | Available |
| `hosa_bicoherence` | Higher-order spectral bicoherence. | Experimental |
| `cross_frequency_coupling` | Comodulogram-based cross-frequency coupling. | Deprecated |

### Signal models, decomposition, and distance

| Feature | Description |
|---|---|
| `ARMA_kalman_filter` | ARMA model coefficients. |
| `EMD` | Empirical mode decomposition into intrinsic mode functions. |
| `Itakura_Distance` | Dynamic-time-warping distance with an Itakura constraint. |

## Important notes

- EEG preprocessing is not performed automatically. Apply appropriate filtering, artifact rejection, referencing, epoching, and normalization before feature extraction.
- Several entropy and decomposition methods require sufficiently long input windows.
- Connectivity features require multichannel data. Some spectral connectivity methods also benefit from dividing the signal into multiple windows with `n_win`.
- Frequency-dependent functions require a correct sampling frequency.
- Feature scales can differ substantially. Standardize features before many machine-learning analyses.
- Experimental or deprecated functions should be validated carefully before research use.

## Documentation and examples

The full documentation includes:

- [Easy beginning](https://scut-eeg-feature.readthedocs.io/en/latest/example/easy_beginning.html)
- [Feature examples](https://scut-eeg-feature.readthedocs.io/en/latest/example/function.html)
- [Functional connectivity](https://scut-eeg-feature.readthedocs.io/en/latest/example/connectivity.html)
- [Phase-amplitude coupling](https://scut-eeg-feature.readthedocs.io/en/latest/example/Phase_amplitude_coupling.html)
- [Amplitude-amplitude coupling](https://scut-eeg-feature.readthedocs.io/en/latest/example/Amplitude_amplitude_coupling.html)
- [Aperiodic analysis](https://scut-eeg-feature.readthedocs.io/en/latest/example/aperodic_analysis.html)

## Contributing

Contributions are welcome. You can help by:

- Reporting bugs or requesting features through [GitHub Issues](https://github.com/didi226/scut_eeg_feature/issues).
- Improving documentation and examples.
- Adding tests for existing features.
- Proposing new EEG or time-series feature functions through pull requests.

When contributing a new feature, use the `compute_<feature_name>` naming convention and include a clear docstring, parameter definitions, output shape, reference, and test.

## Citation

If `scuteegfe` is useful in your research, please cite:

```bibtex
@article{chen2023eeg,
  title   = {An EEG-based attention recognition method: fusion of time domain, frequency domain, and non-linear dynamics features},
  author  = {Chen, Di and Huang, Haiyun and Bao, Xiaoyu and Pan, Jiahui and Li, Yuanqing},
  journal = {Frontiers in Neuroscience},
  volume  = {17},
  pages   = {1194554},
  year    = {2023},
  doi     = {10.3389/fnins.2023.1194554}
}
```
