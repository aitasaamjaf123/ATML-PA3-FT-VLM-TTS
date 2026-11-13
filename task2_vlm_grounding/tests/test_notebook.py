import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[2] / "task2_kaggle_runner.ipynb"


@pytest.mark.skipif(not NB.exists(), reason="notebook not present in this checkout")
def test_notebook_is_valid_and_code_compiles():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for i, c in enumerate(code):
        compile("".join(c["source"]), f"cell{i}", "exec")
    src = "\n".join("".join(c["source"]) for c in code)
    assert "SMOKE_TEST" in src and "build_manifest" in src and "download_task2.py images" in src
