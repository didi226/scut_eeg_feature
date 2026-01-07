:orphan:

Installation
============

Dependencies
------------
* ``python`` (>=3.8)
* ``mne`` (>=1.6)
* ``numpy`` (>=1.21)
* ``scipy`` (>=1.4.0)
* ``scikit-learn`` (>= 1.3.0, <1.7.0)


We require that you use Python 3.10 or higher.
You may choose to install ``scuteegfe`` :ref:`via conda <installation_via_conda>` or :ref:`via pip <installation_via_pip>`.

.. _installation_via_conda:

Installation via Conda
----------------------

To install scuteegfe using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.10 at least
   conda create -n scuteegfe
   conda activate scuteegfe
   conda install -c conda-forge scuteegfe

.. _installation_via_pip:

Installation via Pip
--------------------

To install scuteegfe including all dependencies required to use all features, simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -U scuteegfe



To check if everything worked fine, the following command should not give any error messages:

.. code-block:: bash

   python -c 'import '

To ensure  scuteegfe is up-to-date, run:

.. code-block:: bash

   pip install -U scuteegfe
