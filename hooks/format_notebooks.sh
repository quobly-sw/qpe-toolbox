#!/usr/bin/env bash
# This script converts examples from py:percent to ipynb format
# then runs ruff format on them, using ruff special format of notebooks
# finally convert formatted notebooks back to examples as py:percent
# This is a workaround for current lack of ruff support for py:percent/jupytext
# see https://github.com/astral-sh/ruff/issues/8800

set -euo pipefail

# pre-commit always runs in project root directory
# jupyter_execute is the standard jupyter workdir, already in .gitignore
mkdir -p jupyter_execute

# remove previously existing notebooks
# jupytext would overwrite, but there may be notebooks absent from hook arguments
# we do not want to convert them back
rm -f jupyter_execute/*ipynb

# jupytext working dir is the one of the notebook, need relative  path from examples/
uv run jupytext -q --to ../jupyter_execute//ipynb $@
uv run ruff format jupyter_execute/*ipynb
uv run jupytext -q --to ../examples//py jupyter_execute/*ipynb
