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

def walk_of_types(
    node: ast.AST,
    kinds: tuple[type[ast.AST], ...],
    prune: tuple[type[ast.AST], ...] | None = None,
) -> list[ast.AST]:
    """Return every node under `node` (`node` included) whose exact type is
    in `kinds`, in depth-first pre-order. Unlike :func:`walk_frontier`,
    matches are descended into, except a node whose exact type is in
    `prune`: it is returned if it matches and its children are skipped.
    `node` itself is always descended.
    """

def subtree_hash(node: ast.AST) -> int:
    """``subtree_hashes(node)[id(node)]``, without building the dict."""

def set_parents(node: ast.AST) -> None:
    """Set ``child.parent`` for every node under `node`, visiting parents in
    depth-first pre-order. A node reachable from two parents keeps the last.
    """

def walk_frontier_events(
    node: ast.AST,
    kinds: tuple[type[ast.AST], ...],
    bracket: tuple[type[ast.AST], ...],
) -> list[tuple[int, ast.AST]]:
    """The calls a ``NodeVisitor`` makes descending from `node`.

    ``(0, n)`` is a descendant whose exact type is in `kinds`, dispatched and
    not descended. ``(1, n)`` and ``(2, n)`` surround the descent of a node
    whose exact type is in `bracket`. `node` itself is always descended.
    """

def fix_missing_locations[T: ast.AST](node: T) -> T:
    """``ast.fix_missing_locations`` without recursing once per tree level."""

def subtree_hashes(node: ast.AST) -> dict[int, int]:
    """A structural hash per node under `node`, keyed by ``id(node)``.

    Subtrees with equal ``ast.dump`` output get equal hashes. A leaf value's
    type is mixed in, so ``1``, ``1.0`` and ``True`` hash apart.
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
