# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qpe-toolbox"
project_copyright = "2026, Quobly and Foxconn"
author = "Quobly"
release = "2026, Quobly"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx.ext.autodoc",  # core: import modules, read docstrings
    "sphinx.ext.napoleon",  # NumPy / Google style docstrings
    "sphinx.ext.autosummary",  # auto-generate API stub pages
    "sphinx.ext.viewcode",  # link to highlighted source code
    "autoapi.extension",  # AutoAPI for automatic API documentation
    "myst_nb",  # For notebooks integration
    "sphinx_design",  # For better design blocks
    "sphinx_copybutton",  # For copy buttons in code blocks
    "sphinx.ext.intersphinx",  # Link to other projects documentation
    # "sphinx.ext.extlinks",  # to be added when github repo is done
    # "sphinx.ext.linkcode",  # to be added when github repo is done
]

templates_path = ["_templates"]

# -- Link to other documentation -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Napoleon configuration ---------------------------------------------------
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_attr_annotations = False

# -- AutoAPI configuration ------------------------------------------------

autodoc_typehints = "both"  # or "description"
autoapi_typehints = "description"  # or "signature"

autoapi_dirs = ["../../src/qpe_toolbox"]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "private-members",  # optional, but useful in dev
    "special-members",
    "show-inheritance",
    "show-module-summary",
    "undoc-members",
]

autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_keep_files = True
autoapi_generate_api_docs = True
autoapi_template_dir = "_templates/autoapi"
autoapi_ignore = ["*.ipynb_checkpoints*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "QPE Toolbox"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "hsl(210, 50%, 50%)",
        "color-brand-content": "hsl(210, 50%, 50%)",
    },
    "dark_css_variables": {
        "color-brand-primary": "hsl(210, 50%, 60%)",
        "color-brand-content": "hsl(210, 50%, 60%)",
    },
    "light_logo": "qpe-toolbox.png",
    "dark_logo": "qpe-toolbox.png",
}

pygments_style = "default"  # enable syntax highlighting

html_static_path = ["_static"]
html_css_files = ["my-styles.css"]

# -- Options for notebook execution -------------------------------------------
nb_execution_mode = "off"
myst_heading_anchors = 4
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
