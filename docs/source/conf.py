# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path
from sphinx.locale import _
sys.path.append(str(Path(".").resolve()))
sys.path.insert(0, os.path.abspath('../../src'))

project = 'SCUT EEG Feature'
copyright = '2024, SCUT EEG  Community'
author = 'XiaoYu Bao & Di Chen'
version= '0.4.2'
release = '0.4.2'
extensions = [ 
    'sphinx.ext.napoleon',
    #'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.graphviz',
    'sphinxext.rediraffe',
    'sphinx_design',
    'sphinx_copybutton',
    'autoapi.extension',
    '_extension.gallery_directive',
    '_extension.component_directive',

    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    # 'myst_nb',
    'myst_parser',
    'ablog',
    'jupyter_sphinx',
    'sphinxcontrib.youtube',
    'nbsphinx',
    'numpydoc',
    'sphinx_togglebutton',
    'jupyterlite_sphinx',
    'sphinx_favicon',
    ]

jupyterlite_config = "jupyterlite_config.json"

# source_suffix ={
# '.rst': 'restructuredtext',
# '.ipynb': 'myst-nb',
# '.myst': 'myst-nb',
# }

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mne": ("https://mne.tools/dev", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}
myst_enable_extensions = ["colon_fence" , "substitution"]#"linkify"
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

language = "en"

# -- Ablog options -----------------------------------------------------------
html_show_sphinx = False
blog_path = "examples/blog/index"
blog_authors = {
    "pydata": ("PyData", "https://pydata.org"),
    "jupyter": ("Jupyter", "https://jupyter.org"),
}

togglebutton_hint = str(_("Click to expand"))
togglebutton_hint_hide = str(_("Click to collapse"))

copybutton_selector = ":not(.prompt) > div.highlight pre"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

graphviz_output_format = "svg"

inheritance_graph_attrs = dict(
    rankdir="LR",
    fontsize=14,
    ratio="compress",
)

# autoapi_ignore = [
#     "*HOSA/conventional/*",
# ]



html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"


autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
# autoapi_dirs = ["../../src"]
autoapi_dirs = ["../../src/scuteegfe"]
autoapi_keep_files = False
autoapi_root = "api"
autoapi_member_order = "groupwise"
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = True





# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# nbsphinx_custom_formats = {
#     ".md": ["jupytext.reads", {"fmt": "mystnb"}],
# }
html_theme = "pydata_sphinx_theme"
# html_sourcelink_suffix = ""
# html_last_updated_fmt = ""  # to reveal the build date in the pages meta
html_theme_options = {
    'navigation_depth': 2,
    "header_links_before_dropdown": 4,
        "logo": {
        "text": "SCUT-EEG-Feature Document",
        "image_dark": "_static/logo-dark.svg",
    },
     "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/didi226/scut_eeg_feature",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },     

        ],
        "announcement": "This is a community-supported different kinds of features for EEG or other time series signal . If you'd like to contribute, check out  <a href='https://github.com/didi226/scut_eeg_feature'>our GitHub repository</a>. Your contributions are welcome! ",
        }

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]



htmlhelp_basename = "SEFEAdoc"
html_show_sourcelink = False
html_copy_source = False
html_static_path = ['_static']
# html_css_files = ["style.css"]

