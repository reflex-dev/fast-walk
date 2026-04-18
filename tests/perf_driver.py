"""Driver for perf profiling: one parse, many walks.

Usage:
    maturin develop --release
    perf record -F 2999 --call-graph=dwarf -- \
        taskset -c 2 python tests/perf_driver.py
    perf report --stdio --no-children | head -40
"""

import difflib
from ast import parse
from pathlib import Path
from fast_walk import walk_unordered

node = parse(Path(difflib.__file__).read_text())

for _ in range(200_000):
    walk_unordered(node)
