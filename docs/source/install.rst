:orphan:

Installation
============

Dependencies
------------
* ``python`` (>=3.8)
* ``mne`` (>=1.6)
* ``numpy`` (>=1.21)
* ``scipy`` (>=1.4.0)
* ``scikit-learn`` (>= 1.3.0)


We require that you use Python 3.9 or higher.
You may choose to install ``scuteegfe`` `via pip <#Installation via pip>`_,
or conda.

Installation via Conda
----------------------

To install Sleep-Semantic-Segmentation using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.9 at least
   conda create -n scuteegfe
   conda activate scuteegfe
   conda install -c conda-forge scuteegfe


Installation via Pip
--------------------

To install Sleep-Semantic-Segmentation including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -U scuteegfe

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-connectivity/zipball/main

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import '

sleep-semantic-segmentation works best with the latest stable release of SSSM. To ensure
SSSM is up-to-date, run:

.. code-block:: bash

   pip install --user -U scuteegfe
