"""`walk_frontier` and `walk_frontier_edges` must match a pure-Python reference:
the nodes `ast.NodeVisitor.generic_visit` dispatches to `visit_<kind>`, given
every other node falls through to the generic descent."""

from __future__ import annotations

import ast
import gc
import sys
import textwrap

import pytest

from fast_walk import walk_frontier, walk_frontier_edges

SOURCE = textwrap.dedent("""
    import x as y
    def f(a, b=g(h.i)):
        c = [k(l) for l in m.n.o]
        if c:
            return p.q(r, *s, t=u.v)
        class W(X.Y):
            z: Z = w.call()
    lambda: q.attr
""")

KINDS = [
    (ast.Call,),
    (ast.Attribute,),
    (ast.Call, ast.Attribute),
    (ast.FunctionDef, ast.Name),
    (ast.Load,),
    (),
]


def reference(node: ast.AST, kinds: tuple[type, ...]):
    found = []

    def descend(parent: ast.AST):
        for field, value in ast.iter_fields(parent):
            items = value if isinstance(value, list) else [value]
            for index, child in enumerate(items):
                if not isinstance(child, ast.AST):
                    continue
                if type(child) in kinds:
                    found.append((
                        child,
                        parent,
                        field,
                        index if isinstance(value, list) else -1,
                    ))
                else:
                    descend(child)

    descend(node)
    return found


def identities(edges):
    return [(id(n), id(p), f, i) for n, p, f, i in edges]


@pytest.mark.parametrize("kinds", KINDS)
def test_edges_match_the_generic_visit_order(kinds):
    tree = ast.parse(SOURCE)
    assert identities(walk_frontier_edges(tree, kinds)) == identities(
        reference(tree, kinds)
    )


@pytest.mark.parametrize("kinds", KINDS)
def test_frontier_is_the_edge_nodes(kinds):
    tree = ast.parse(SOURCE)
    assert [id(n) for n in walk_frontier(tree, kinds)] == [
        id(edge[0]) for edge in walk_frontier_edges(tree, kinds)
    ]


def test_the_root_is_never_included():
    tree = ast.parse("f(g())")
    call = tree.body[0].value
    assert [ast.unparse(n) for n in walk_frontier(call, (ast.Call,))] == ["g()"]


def test_a_match_is_not_descended():
    tree = ast.parse("a.b.c.d")
    assert [ast.unparse(n) for n in walk_frontier(tree, (ast.Attribute,))] == [
        "a.b.c.d"
    ]


def test_a_deep_chain_does_not_recurse():
    tree = ast.parse("x" + ".y" * 20_000)
    assert [type(n) for n in walk_frontier(tree, (ast.Name,))] == [ast.Name]


def test_kinds_must_be_types():
    with pytest.raises(TypeError):
        walk_frontier(ast.parse("x"), ("Name",))


def test_refcount_neutral():
    tree = ast.parse(SOURCE)
    nodes = list(ast.walk(tree))
    gc.collect()
    before = [sys.getrefcount(n) for n in nodes]
    for _ in range(50):
        walk_frontier(tree, (ast.Call, ast.Attribute))
        walk_frontier_edges(tree, (ast.Call, ast.Attribute, ast.Name))
    gc.collect()
    assert [sys.getrefcount(n) for n in nodes] == before


def test_fields_are_read_by_name():
    call = ast.Call(lineno=1, func=ast.Name(id="f", ctx=ast.Load()), args=[])
    assert [(type(n).__name__, f, i) for n, _, f, i in walk_frontier_edges(call, (ast.Name,))] == [
        ("Name", "func", -1)
    ]
