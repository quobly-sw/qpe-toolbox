#!/usr/bin/env python3

import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).parents[1] / "examples"
EXAMPLES = sorted(path.name for path in EXAMPLES_DIR.glob("*.py"))

# non-interactive matplotlib backend, silence tqdm
EXAMPLE_ENV = os.environ | {"MPLBACKEND": "agg", "TQDM_DISABLE": "1"}


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    """Test that examples run without errors."""
    # run from EXAMPLES_DIR to keep the same paths as within the notebooks
    result = subprocess.run(
        [sys.executable, example],
        cwd=EXAMPLES_DIR,
        env=EXAMPLE_ENV,
        text=True,
        check=False,
    )
    assert result.returncode == 0


if __name__ == "__main__":
    for ex in EXAMPLES:
        test_example_runs(ex)
