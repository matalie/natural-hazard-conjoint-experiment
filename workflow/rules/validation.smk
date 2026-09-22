"""Repository validation rules."""

from pathlib import Path

rule check:
    input:
        list(map(str, Path(SRC).glob("*.py"))),
        list(map(str, Path("tests").glob("*.py"))),
        list(map(str, Path("workflow/scripts").glob("*.py"))),
        list(map(str, Path("tools").glob("*.py"))),
        "config/config.yaml",
    output: report=f"{ROOT}/validation/tests.txt"
    script: "../scripts/check.py"