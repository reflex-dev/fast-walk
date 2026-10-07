import ast

def walk_dfs(node: ast.AST) -> list[ast.AST]:
    """Return every descendant of `node` (including `node` itself) in strict
    depth-first pre-order.

    Semantically equivalent to ``list(ast.walk(node))`` but much faster.
    Use :func:`walk_unordered` if traversal order doesn't matter — it's
    faster still.
    """

def walk_unordered(node: ast.AST) -> list[ast.AST]:
    """Return every descendant of `node` (including `node` itself) in an
    implementation-defined order.

    The set of returned nodes is identical to :func:`walk_dfs` and to
    :func:`ast.walk`; only the visit order differs. Since :func:`ast.walk`
    makes no ordering guarantee, this is a drop-in replacement wherever
    the caller does not depend on DFS order.

    Uses batched stack draining with L1 prefetch hints to hide the
    cache-miss latency of scattered ``PyDictKeysObject`` loads — roughly
    25% faster than :func:`walk_dfs` on real Python source.
    """

def walk_frontier(
    node: ast.AST, kinds: tuple[type[ast.AST], ...]
) -> list[ast.AST]:
    """Return the descendants of `node` whose exact type is in `kinds`, in
    depth-first pre-order, without descending below any match. `node`
    itself is never included.

    These are the nodes ``ast.NodeVisitor.generic_visit`` dispatches to a
    ``visit_<kind>`` method when every other node falls through to the
    generic descent.
    """

def walk_frontier_edges(
    node: ast.AST, kinds: tuple[type[ast.AST], ...]
) -> list[tuple[ast.AST, ast.AST, str, int]]:
    """Like :func:`walk_frontier`, but each match comes as
    ``(node, parent, field, index)``: it was read from ``parent.<field>``
    when ``index`` is ``-1``, else from ``parent.<field>[index]``.
    """

def walk(node: ast.AST) -> list[ast.AST]:
    """Deprecated. Use :func:`walk_dfs` for explicit depth-first order or
    :func:`walk_unordered` for the faster order-agnostic variant.

    Emits a :class:`DeprecationWarning` once per process on first call and
    then delegates to :func:`walk_dfs`.
    """

def _walk_count(node: ast.AST) -> int:
    """Benchmarking-only. Traverse the AST and return the node count without
    materializing a result list.
    """
