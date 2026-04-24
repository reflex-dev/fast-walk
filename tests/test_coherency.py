"""Coherency tests — `walk_dfs`, `walk_unordered`, and `walk` must agree with
each other and with the stdlib `ast.walk` across a range of inputs.

Each helper produces the same SET of AST-node identities (multiset actually —
shared singletons like `ast.Load()` legitimately appear multiple times).
They may differ in visit order. These tests pin down what must be invariant
regardless of order.
"""

from __future__ import annotations

import ast
import textwrap
import warnings
from collections import Counter

import pytest

from fast_walk import walk_dfs, walk_unordered
import fast_walk


SOURCES: dict[str, str] = {
    "empty": "",
    "single_stmt": "x = 1",
    "just_a_literal": "42\n",
    "functions": textwrap.dedent("""
        def f(x, y, /, z, *args, kw=1, **kwargs):
            return x + y * z

        async def g():
            async for i in it():
                async with ctx() as c:
                    yield i
    """),
    "classes": textwrap.dedent("""
        class A:
            x: int = 0

            def m(self, *a, **k) -> int:
                return self.x

        class B(A, metaclass=type):
            ...
    """),
    "comprehensions": textwrap.dedent("""
        squares = [x * x for x in range(10) if x % 2]
        d = {k: v for k, v in pairs if k}
        g = (x async for x in aiter() if x)
        s = {a + b for a in xs for b in ys}
    """),
    "control_flow": textwrap.dedent("""
        for i in range(10):
            if i % 2 == 0:
                continue
            elif i > 7:
                break
            else:
                pass
        try:
            risky()
        except (ValueError, TypeError) as e:
            handle(e)
        except Exception:
            raise
        finally:
            cleanup()
        while cond:
            do()
        with open('x') as f, open('y') as g:
            pass
        match obj:
            case [1, 2, *rest]:
                pass
            case {"k": v, **rest}:
                pass
            case Point(x=0, y=0):
                pass
            case _:
                pass
    """),
    "deeply_nested": "a = " + "[" * 50 + "1" + "]" * 50,
    "imports_and_decorators": textwrap.dedent("""
        import a, b.c
        from d.e import f as g, h
        from . import i

        @decorator
        @deco2(arg)
        def decorated():
            pass

        @dataclass
        class Point:
            x: int
            y: int = 0
    """),
    "fstrings_and_strings": textwrap.dedent("""
        s = f"hello {name!r:>10}"
        b = b"bytes"
        multiline = '''
        a
        b
        '''
    """),
    "type_params_pep695": textwrap.dedent("""
        def generic[T, U: int, *Ts, **P](x: T) -> T:
            return x

        class Container[T]:
            def __init__(self, value: T) -> None:
                self.value = value

        type Alias[T] = list[T]
    """),
    # Wider / deeper sources to exercise traversal on bigger trees. The
    # smaller sources above catch most syntactic edge cases; these ones
    # are here to surface scale-sensitive bugs — unbounded stack growth,
    # quadratic behaviour, cycle-like regressions — that only appear once
    # the node count is large enough.
    "many_small_statements": "\n".join(f"x{i} = {i}" for i in range(500)),
    "wide_function_body": textwrap.dedent("""
        def big():
    """) + "\n".join(f"    y{i} = f({i}) + g({i})" for i in range(300)),
    "deeply_nested_calls": "f(" * 100 + "x" + ")" * 100,
    "deeply_nested_attributes": "a" + ".b" * 200,
    "deeply_nested_binops": " + ".join(f"x{i}" for i in range(300)),
}


def _multiset(nodes) -> Counter[int]:
    """Collapse a walk result to a Counter of id()s.

    AST walks legitimately contain the same node object multiple times
    (shared singletons like `ast.Load()`), so equality of sets isn't
    enough — we need equality of multisets.
    """
    return Counter(id(n) for n in nodes)


@pytest.fixture(params=sorted(SOURCES.keys()))
def tree(request) -> ast.AST:
    return ast.parse(SOURCES[request.param])


def test_walk_dfs_matches_ast_walk(tree: ast.AST):
    """walk_dfs must produce the same multiset as ast.walk."""
    assert _multiset(walk_dfs(tree)) == _multiset(ast.walk(tree))


def test_walk_unordered_matches_ast_walk(tree: ast.AST):
    """walk_unordered must produce the same multiset as ast.walk, even
    though the order is not guaranteed."""
    assert _multiset(walk_unordered(tree)) == _multiset(ast.walk(tree))


def test_walk_dfs_and_unordered_agree(tree: ast.AST):
    """The two fast-path implementations must agree on the multiset of
    visited nodes."""
    assert _multiset(walk_dfs(tree)) == _multiset(walk_unordered(tree))


def test_walk_dfs_visits_root_first(tree: ast.AST):
    """Pre-order DFS: the first node in the result is always the root."""
    result = walk_dfs(tree)
    assert result[0] is tree


def test_walk_unordered_visits_root_first(tree: ast.AST):
    """The batched implementation currently pops the seed root first
    regardless of order within later batches. Pin that behaviour so a
    future refactor doesn't silently stop returning the input as the
    first element."""
    result = walk_unordered(tree)
    assert result[0] is tree


def test_walk_dfs_is_deterministic(tree: ast.AST):
    """Repeated calls must return identical sequences — no thread-local
    or cached state should leak between invocations."""
    a = [id(n) for n in walk_dfs(tree)]
    b = [id(n) for n in walk_dfs(tree)]
    c = [id(n) for n in walk_dfs(tree)]
    assert a == b == c


def test_walk_unordered_is_deterministic(tree: ast.AST):
    """The *unordered* variant still has a deterministic implementation.
    Non-determinism here would be a bug (e.g. uninitialised scratch)."""
    a = [id(n) for n in walk_unordered(tree)]
    b = [id(n) for n in walk_unordered(tree)]
    assert a == b


def test_walk_dfs_pre_order_property(tree: ast.AST):
    """For each AST node with child AST nodes, the parent must appear
    before any of its descendants in walk_dfs order. This is the
    defining property of pre-order DFS."""
    order = {id(n): i for i, n in enumerate(walk_dfs(tree))}

    for node in ast.walk(tree):
        parent_index = order[id(node)]
        for child in ast.iter_child_nodes(node):
            # Shared singletons (Load/Store/...) may appear earlier via
            # a different parent; only require that *some* occurrence of
            # the child lies after this parent. Since walks return one
            # entry per visit, we just require parent_index < child_index
            # for child nodes reachable via this parent.
            child_index = order.get(id(child))
            assert child_index is not None
            assert parent_index < child_index, (
                f"{type(node).__name__} at {parent_index} should precede "
                f"its child {type(child).__name__} at {child_index}"
            )


def test_walk_deprecated_alias_matches_dfs(tree: ast.AST):
    """The deprecated `walk` entry point must return exactly the same
    sequence as walk_dfs. We swallow the DeprecationWarning since the
    point of this test is equivalence, not emission."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = [id(n) for n in fast_walk.walk(tree)]
    dfs = [id(n) for n in walk_dfs(tree)]
    assert legacy == dfs


def test_walk_returns_list():
    """Both entry points must return a built-in list (not a generator or
    custom iterable), matching the documented return type."""
    tree = ast.parse(SOURCES["functions"])
    assert type(walk_dfs(tree)) is list
    assert type(walk_unordered(tree)) is list


def test_non_tree_leaf_inputs():
    """walk_dfs/walk_unordered on a single leaf node (no descendants)
    should return exactly [node]."""
    node = ast.parse("1").body[0].value  # an ast.Constant
    assert walk_dfs(node) == [node]
    assert walk_unordered(node) == [node]


def test_walk_dfs_order_is_stable_across_independent_trees():
    """Two parses of the same source must produce isomorphic walk
    sequences — same type sequence in the same order. This catches
    any accidental dependence on node identity/address during
    traversal decisions."""
    src = SOURCES["control_flow"]
    a_types = [type(n).__name__ for n in walk_dfs(ast.parse(src))]
    b_types = [type(n).__name__ for n in walk_dfs(ast.parse(src))]
    assert a_types == b_types


def test_matches_stdlib_for_sizable_real_file():
    """End-to-end coherency on a real, nontrivial module."""
    import difflib
    from pathlib import Path

    tree = ast.parse(Path(difflib.__file__).read_text())
    expected = _multiset(ast.walk(tree))
    assert _multiset(walk_dfs(tree)) == expected
    assert _multiset(walk_unordered(tree)) == expected


@pytest.mark.parametrize("walk_fn", [walk_dfs, walk_unordered])
def test_parent_back_references_do_not_inflate_walk(walk_fn, tree: ast.AST):
    """Decorating nodes with `.parent` back-references is a common AST-
    transformer pattern. `ast.walk` never follows non-`_fields` keys, so
    the back-refs are invisible to it — fast_walk must match. Under the
    old "scan every __dict__ value" implementation this would either
    infinite-loop or silently multiply the result; today it's guarded by
    the per-type `_fields`-length bound."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # pyright: ignore[reportAttributeAccessIssue]

    assert _multiset(walk_fn(tree)) == _multiset(ast.walk(tree))


@pytest.mark.parametrize("walk_fn", [walk_dfs, walk_unordered])
def test_self_reference_does_not_inflate_walk(walk_fn):
    """Minimal cycle: a node attaches a reference to itself via a non-
    `_fields` attribute. Must terminate and not double-count."""
    tree = ast.parse("x = 1")
    tree.body[0].self_ref = tree.body[0]  # pyright: ignore[reportAttributeAccessIssue]

    assert _multiset(walk_fn(tree)) == _multiset(ast.walk(tree))


@pytest.mark.parametrize("walk_fn", [walk_dfs, walk_unordered])
def test_non_fields_ast_attribute_is_ignored(walk_fn):
    """A user-attached AST reference outside `_fields` must not leak
    into the walk result — not even when there's no cycle. Keeps us
    strictly equivalent to `ast.walk`."""
    tree = ast.parse("x = 1")
    sibling = ast.parse("y = 2").body[0]
    tree.body[0].extra = sibling  # pyright: ignore[reportAttributeAccessIssue]

    result = walk_fn(tree)
    assert _multiset(result) == _multiset(ast.walk(tree))
    assert id(sibling) not in {id(n) for n in result}


def test_parent_back_references_on_real_file():
    """Scale check for the cycle fix: every node in a real module gets
    a `.parent` back-reference, and both walks must still match
    `ast.walk` exactly. Under a regression this would exhaust memory."""
    import difflib
    from pathlib import Path

    tree = ast.parse(Path(difflib.__file__).read_text())
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # pyright: ignore[reportAttributeAccessIssue]

    expected = _multiset(ast.walk(tree))
    assert _multiset(walk_dfs(tree)) == expected
    assert _multiset(walk_unordered(tree)) == expected


