
.. _Contribute:

Contribute to SEFEA
********************

There are many ways to contribute to SEFEA: reporting bugs, adding new functions, improving the documentation, etc...

If you like SEFEA, you can also consider `buying the developers a headband from HNNK <https://item.taobao.com/item.htm?spm=a21n57.1.item.2.76f05263QxvUVm&priceTId=2150407317212886249531433e9816&utparam=%7B%22aplus_abtest%22:%22b85e6cf8b990cb19de672e9c8381c9e4%22%7D&id=745006412856&ns=1&abbucket=15>`_!

Code guidelines
----------------

Before starting new code, we highly recommend opening an issue on `GitHub <https://github.com/didi226/scut_eeg_feature>`_ to discuss potential changes.

* Please use standard `pep8 <https://pypi.python.org/pypi/pep8>`_ and `flake8 <http://flake8.pycqa.org/>`_ Python style guidelines. To test that your code complies with those, you can run:

  .. code-block:: bash

     $ flake8

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* When adding new functions, make sure that they are **generalizable to various situations**.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

  .. code-block:: bash

     $ pytest

Checking and building documentation
------------------------------------

SSS's documentation (including docstring in code) uses ReStructuredText format,
see `Sphinx documentation <http://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
follows the `Pydata-Sphinx-Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html>`_.

All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings. If you want to test the documentation locally, you will need to install the following packages:

.. code-block:: bash

  $ pip install --upgrade sphinx pydata-sphinx-theme

and then within the ``scut_eeg_feature/docs`` directory do:

.. code-block:: bash

  $ make html
