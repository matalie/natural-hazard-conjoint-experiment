from pathlib import Path
import subprocess
import sys


def main(s):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"), "-q"], cwd=root,
                             capture_output=True, text=True)
    text = result.stdout + result.stderr
    print(text)
    if result.returncode:
        raise RuntimeError("Tests failed; see the output above.")
    Path(s.output.report).parent.mkdir(parents=True, exist_ok=True)
    Path(s.output.report).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(globals()["snakemake"])
