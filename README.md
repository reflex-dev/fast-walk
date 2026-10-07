# fast-walk

A fast reimplementation of Python's `ast.walk`, written in Rust.

## Installation

```bash
pip install fast-walk
```

## Usage

Two public entry points, depending on whether you care about traversal order:

```python
import ast
from fast_walk import walk_dfs, walk_unordered

tree = ast.parse("def f(x): return x + 1")

# Strict depth-first pre-order.
for node in walk_dfs(tree):
    ...

# Implementation-defined order; same node set, faster.
# ast.walk itself makes no ordering guarantee, so this is a drop-in for most code.
for node in walk_unordered(tree):
    ...
```

### Which one to use

- **`walk_unordered`** — default choice. Same set of nodes as `ast.walk`
  but with better cache behavior (batched dict-metadata prefetching).
  Roughly 25% faster than `walk_dfs` on real Python source.
- **`walk_dfs`** — pick this only if your code actually depends on
  depth-first pre-order visitation. `ast.walk` does not document an order,
  so most callers can safely use `walk_unordered`.

### Visiting only some node types

`walk_frontier(node, kinds)` returns the descendants whose exact type is in
`kinds`, in depth-first pre-order, without descending below a match. That is
the set `ast.NodeVisitor.generic_visit` hands to `visit_<kind>` methods, so a
visitor whose only methods are `visit_<kind>` can descend with:

```python
def generic_visit(self, node):
    for child in walk_frontier(node, (ast.Call, ast.Attribute)):
        self.visit(child)
```

`walk_frontier_edges` returns `(node, parent, field, index)` tuples instead
(`index` is `-1` for a non-list field), which is what a `NodeTransformer`
needs to put a replacement back.

### Other helpers

- `walk_of_types(node, kinds, prune=None)` — every node of the given exact
  types, in pre-order, descending into matches but not below a `prune` type.
- `walk_frontier_events(node, kinds, bracket)` — the frontier walk as
  `(event, node)` pairs, with enter/leave events around `bracket` types, for a
  visitor that does work around its generic descent.
- `set_parents(node)` — `child.parent = node` for the whole tree.
- `fix_missing_locations(node)` — `ast.fix_missing_locations` without
  recursion.
- `subtree_hashes(node)` — `{id(n): hash}` such that equal hashes mean equal
  `ast.dump` output (leaf types included, so `1` and `True` differ).
  `subtree_hash(node)` is the root's entry alone.

These, and the frontier walks, read each child by its `_fields` name, so
they also work on nodes a transformer built with keyword arguments in
another order. `walk_dfs` and `walk_unordered` read fields by position,
which is only guaranteed for parsed trees.

## Performance

Benchmark on CPython 3.13, walking the AST of `difflib.py` (~2000 lines,
~4300 unique AST nodes), best-of-N run pinned to a single CPU with the
`performance` governor:

| implementation             | min time | relative |
| -------------------------- | -------- | -------- |
| `ast.walk` (stdlib)        | ~1.9 ms  | 1×       |
| pure-Python equivalent     | ~400 µs  | ~5×      |
| `fast_walk.walk_dfs`       | ~5.6 µs  | ~340×    |
| `fast_walk.walk_unordered` | ~4.3 µs  | ~440×    |

Both `fast_walk` entry points are semantically equivalent to
`list(ast.walk(node))` — they return the same set of AST nodes. They
differ only in visit order. User-attached attributes outside `_fields`
(e.g. a `.parent` back-reference set by an AST transformer) are
ignored, matching `ast.walk`'s behaviour.

## Development

### Prerequisites

- Rust (latest stable)
- Python 3.13+
- [maturin](https://github.com/PyO3/maturin)

### Building from source

```bash
pip install maturin

# Iterative builds (debug profile):
maturin develop

# Optimized builds for benchmarking:
maturin develop --release
```

### Running the tests and benchmarks

```bash
# Correctness + refcount leak tests:
pytest tests/test_refcount.py

# Benchmarks (codspeed, walltime mode):
pytest tests/benchmarks.py --codspeed
```

## License

MIT

## Links

- [Repository](https://github.com/reflex-dev/fast-walk)
