# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import ast
import os

# -- Source tree access -------------------------------------------------------
# conf.py is docs/source/conf.py, so the repository root sits three levels up.
# The package is read from source with `ast` and never imported
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SRC = os.path.join(_REPO_ROOT, "src")
_GITHUB_BLOB = "https://github.com/quobly-sw/qpe-toolbox/blob/main"


# Parsed syntax tree of a source file.
def _parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


# Source file of a dotted module name, either `pkg/mod.py` or
# `pkg/mod/__init__.py`, or None when neither exists.
def _module_file(module):
    rel = module.replace(".", os.sep)
    for candidate in (f"{rel}.py", os.path.join(rel, "__init__.py")):
        path = os.path.join(_SRC, candidate)
        if os.path.isfile(path):
            return path
    return None


# Names a syntax node binds: a def/class name, or the targets of an assignment.
def _bound_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


project = "qpe-toolbox"
project_copyright = "2026, Quobly and Foxconn"
author = "Quobly"
# Read __version__ from the same file `[tool.hatch.version]` derives version
release = next(
    ast.literal_eval(node.value)
    for node in _parse(_module_file("qpe_toolbox")).body
    if "__version__" in _bound_names(node)
)
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


# First and last line of `qualname` in `path`, or None when it is not defined
# there. Covers functions, classes, methods and data (EXACT, __version__) alike,
# and starts the range at the first decorator so properties read correctly.
def _find_definition(path, qualname):
    node = _parse(path)
    for part in qualname.split("."):
        node = next(
            (c for c in ast.iter_child_nodes(node) if part in _bound_names(c)), None
        )
        if node is None:
            return None
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)]), node.end_lineno


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    path = _module_file(info["module"])
    if path is None:
        return None
    rel = os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")
    # A name only re-exported by this module has no definition to point at, so
    # the link degrades to the file rather than to unrelated line numbers.
    span = _find_definition(path, info["fullname"]) if info["fullname"] else None
    anchor = ""
    if span:
        start, end = span
        anchor = f"#L{start}-L{end}" if end and end != start else f"#L{start}"
    return f"{_GITHUB_BLOB}/{rel}{anchor}"


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
autoapi_keep_files = False  # no cache is more convenient for local checks
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
