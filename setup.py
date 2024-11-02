import setuptools
from setuptools import find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="scuteegfe",
    version="0.0.2",
    author="bkxcyu",
    author_email="bkxcyu@gmail.com",
    description="EEG Signal Feature Exacting, a wrapper of mne_features",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={"": "src"},
    install_requires=[
        'requests',
        'importlib-metadata; python_version >= "3.8"',
        'einops',
        'mne_features',
        'EMD-signal',
        'scipy',
        'PyWavelets',
        'pyts',
        'antropy',
        'pyentrp',
        'tftb',
        'statsmodels',
        'scipy',
        'emd-signal',
        'pyentrp',
        'nolds',
        'pactools',
        'EntropyHub',
'pykalman',
'mne_connectivity',
'spkit>= 0.0.9.6.8',
'fooof',
'nilearn'
],
)
