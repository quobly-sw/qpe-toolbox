# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import ast
import importlib
import importlib.metadata
import inspect
import os

project = "qpe-toolbox"
project_copyright = "2026, Quobly and Foxconn"
author = "Quobly"
release = importlib.metadata.version("qpe_toolbox")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # NumPy / Google style docstrings
    "sphinx.ext.autosummary",  # auto-generate API stub pages
    "sphinx.ext.linkcode",  # [source] buttons linking to GitHub
    "autoapi.extension",  # AutoAPI for automatic API documentation
    "myst_nb",  # For notebooks integration
    "sphinx_design",  # For better design blocks
    "sphinx_copybutton",  # For copy buttons in code blocks
    "sphinx.ext.intersphinx",  # Link to other projects documentation
    "sphinx.ext.extlinks",
]

# -- Link to other documentation -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Napoleon configuration ---------------------------------------------------
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_attr_annotations = False

# -- Linkcode (GitHub source links) ------------------------------------------


# Line of the module-level or class-level assignment to `qualname`, or None.
# Used for data (EXACT, __version__, class attributes) that have no code object.
def _find_assignment_line(mod, qualname):
    try:
        body = ast.parse(inspect.getsource(mod)).body
    except Exception:
        return None
    *parents, name = qualname.split(".")
    for parent in parents:
        body = next(
            (n.body for n in body if isinstance(n, ast.ClassDef) and n.name == parent),
            None,
        )
        if body is None:
            return None
    for node in body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.lineno
    return None


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        mod = importlib.import_module(info["module"])
    except Exception:
        return None
    obj = mod
    try:
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        if isinstance(obj, property):
            obj = obj.fget
        lines, start = inspect.getsourcelines(obj)
        lineno = f"#L{start}-L{start + len(lines) - 1}"
    except Exception:
        line = _find_assignment_line(mod, info["fullname"])
        lineno = f"#L{line}" if line else ""
    # Resolve the module's real file so package __init__.py maps correctly.
    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        return None
    src_root = os.path.dirname(
        os.path.dirname(importlib.import_module("qpe_toolbox").__file__)
    )
    rel = os.path.relpath(mod_file, src_root)
    return f"https://github.com/quobly-sw/qpe-toolbox/blob/main/src/{rel}{lineno}"


# -- External links -------------------------------------------------------
extlinks = {
    "quimb": ("https://quimb.readthedocs.io/en/latest/%s", "quimb %s"),
    "quimb-api": (
        "https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/index.html"
        "#quimb.tensor.%s",
        "%s",
    ),
    "numpy-api": (
        "https://numpy.org/doc/stable/reference/generated/numpy.%s.html",
        "numpy.%s",
    ),
    "numpy-random": (
        "https://numpy.org/doc/stable/reference/random/%s.html",
        "numpy.random.%s",
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/%s",
        "%s",
    ),
    "matplotlib-api": (
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.%s.html",
        "matplotlib.%s",
    ),
    "pyscf-api": (
        "https://pyscf.org/pyscf_api_docs/pyscf.%s",
        "pyscf.%s",
    ),
    "openfermion-ops": (
        "https://quantumai.google/reference/python/openfermion/ops/%s",
        "openfermion.ops.%s",
    ),
    "cotengra-api": (
        "https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/index.html"
        "#cotengra.%s",
        "cotengra.%s",
    ),
    "qiskit-api": (
        "https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.%s",
        "qiskit.%s",
    ),
}

# -- AutoAPI configuration ------------------------------------------------

autoapi_typehints = "description"  # or "signature"

autoapi_dirs = ["../../src/qpe_toolbox"]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "undoc-members",
]

autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_keep_files = True
autoapi_generate_api_docs = True
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
    "light_logo": "qpe-toolbox_logo.png",
    "dark_logo": "qpe-toolbox_logo.png",
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


# -- AutoAPI member filtering ---------------------------------------------------
# Document qpe_toolbox.__version__, which AutoAPI skips as private.
def keep_package_version(app, what, name, obj, skip, options):
    if name == "qpe_toolbox.__version__":
        return False
    return None


def setup(app):
    app.connect("autoapi-skip-member", keep_package_version)
