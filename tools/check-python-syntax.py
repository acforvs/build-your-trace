#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "benchmarks").glob("*.py")) + sorted((root / "tools").glob("*.py"))
for path in paths:
    compile(path.read_text(), str(path.relative_to(root)), "exec")
print(f"Validated Python syntax for {len(paths)} files.")
