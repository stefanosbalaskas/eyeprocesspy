from pathlib import Path
import runpy
import pytest

EXAMPLES = sorted((Path(__file__).parents[1]/"examples").glob("irt_*.py"))

@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p:p.name)
def test_irt_examples_execute(path):
    runpy.run_path(str(path), run_name="__main__")
