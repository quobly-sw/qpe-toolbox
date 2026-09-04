#!/usr/bin/env bash
# This script converts examples from py:percent to ipynb format
# then runs ruff format on them, using ruff special format of notebooks
# finally convert formatted notebooks back to examples as py:percent
# This is a workaround for current lack of ruff support for py:percent/jupytext
# see https://github.com/astral-sh/ruff/issues/8800

set -euo pipefail

# all the paths below are relative to the repo root, so this script can be
# called from any directory
root=$(git rev-parse --show-toplevel)  # fails the script when called outside the repo
cd "$root"

# if given arguments, format only them
# without argument, format all notebooks
if (($#)); then
    examples=("$@")
else
    mapfile -t examples < <(git ls-files 'examples/*.py')
fi

before=$(sha256sum "${examples[@]}")

# jupyter_execute is the standard jupyter workdir, already in .gitignore
mkdir -p jupyter_execute

# remove previously existing notebooks
# jupytext would overwrite, but a renamed or deleted example may leave one
# behind, we do not want to convert it back
rm -f jupyter_execute/*ipynb

# jupytext working dir is the one of the notebook, need relative  path from examples/
uv run jupytext -q --to ../jupyter_execute//ipynb "${examples[@]}"
uv run ruff format jupyter_execute/*ipynb
uv run jupytext -q --to ../examples//py jupyter_execute/*ipynb

# fail if the round-trip rewrote the examples, so this doubles as a CI check:
# pre-commit detects modified files by itself, a direct caller needs the status
after=$(sha256sum "${examples[@]}")
[[ "$before" == "$after" ]]
