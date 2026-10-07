"""`walk_of_types`, `fix_missing_locations` and `subtree_hashes` read each
node's children by `_fields` name, so they must agree with the stdlib on
parsed trees and on nodes a transformer built in another attribute order."""

from __future__ import annotations

import ast
import copy
import gc
import sys
import textwrap

import pytest

from fast_walk import fix_missing_locations, subtree_hashes, walk_of_types

SOURCE = textwrap.dedent("""
    import os
    from x import y as z
    def f(a: int = 1, *b, c=2.0, **d) -> str:
        e = [g(h) for h in i if h]
        if e and not a:
            return k.l.m(n, *o, p=q)
        with r() as s, t:
            del u[v:w]
    class C(D, metaclass=E):
        x: int = True
        async def m(self):
            async for a in b:
                await c
            match a:
                case [1, *rest] | [2, *rest]:
                    pass
                case {"k": v, **kw}:
                    pass
    lambda q=None: f"{q!r:>{w}}"
""")


def pre_order(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from pre_order(child)


@pytest.mark.parametrize(
    "kinds",
    [(ast.Name,), (ast.Call, ast.Attribute), (ast.Module,), (ast.Load,), ()],
)
def test_walk_of_types_is_the_filtered_pre_order(kinds):
    tree = ast.parse(SOURCE)
    assert [id(n) for n in walk_of_types(tree, kinds)] == [
        id(n) for n in pre_order(tree) if type(n) in kinds
    ]


def test_walk_of_types_reads_fields_by_name():
    call = ast.Call(lineno=1, func=ast.Name(id="f", ctx=ast.Load()), args=[])
    assert [n.id for n in walk_of_types(call, (ast.Name,))] == ["f"]


def stripped(source: str) -> ast.Module:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.expr, ast.stmt)) and len(node._fields) % 2:
            for name in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, name):
                    delattr(node, name)
    return tree


def locations(tree: ast.AST):
    return [
        tuple(getattr(n, name, "-") for name in n._attributes) for n in pre_order(tree)
    ]


def test_fix_missing_locations_matches_the_stdlib():
    ours, theirs = stripped(SOURCE), stripped(SOURCE)
    assert fix_missing_locations(ours) is ours
    ast.fix_missing_locations(theirs)
    assert locations(ours) == locations(theirs)
    compile(ours, "<test>", "exec")


def built() -> ast.Module:
    return ast.Module(
        body=[
            ast.Expr(
                value=ast.Call(
                    lineno=3,
                    func=ast.Name(id="f", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            )
        ],
        type_ignores=[],
    )


def test_fix_missing_locations_fills_a_built_tree():
    ours, theirs = built(), built()
    fix_missing_locations(ours)
    ast.fix_missing_locations(theirs)
    assert locations(ours) == locations(theirs)
    assert ours.body[0].value.func.lineno == 3


def test_fix_missing_locations_on_a_deep_chain():
    tree = ast.parse("x" + ".y" * 20_000)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            del node.end_lineno
    fix_missing_locations(tree)
    assert tree.body[0].value.end_lineno == 1


def pairs(left: ast.AST, right: ast.AST):
    return [(a, b) for a in pre_order(left) for b in pre_order(right)]


def test_subtree_hashes_agree_with_ast_dump():
    old = ast.parse(SOURCE)
    new = copy.deepcopy(old)
    new.body[2].body[0].value.elt.func.id = "changed"
    old_hashes, new_hashes = subtree_hashes(old), subtree_hashes(new)
    for a, b in pairs(old, new):
        assert (old_hashes[id(a)] == new_hashes[id(b)]) == (
            ast.dump(a) == ast.dump(b)
        ), ast.dump(a)


@pytest.mark.parametrize(
    ("left", "right"), [("x = 1", "x = True"), ("x = 1", "x = 1.0"), ("x = 0", "x = False")]
)
def test_subtree_hashes_tell_equal_hashing_constants_apart(left, right):
    a, b = ast.parse(left), ast.parse(right)
    assert subtree_hashes(a)[id(a)] != subtree_hashes(b)[id(b)]


def test_subtree_hashes_ignore_dict_order():
    built = ast.Call(lineno=1, keywords=[], func=ast.Name(id="f", ctx=ast.Load()), args=[])
    parsed = ast.parse("f()").body[0].value
    assert subtree_hashes(built)[id(built)] == subtree_hashes(parsed)[id(parsed)]


def test_refcount_neutral():
    tree = ast.parse(SOURCE)
    nodes = list(ast.walk(tree))
    gc.collect()
    before = [sys.getrefcount(n) for n in nodes]
    for _ in range(50):
        walk_of_types(tree, (ast.Name, ast.Call))
        subtree_hashes(tree)
        fix_missing_locations(tree)
    gc.collect()
    assert [sys.getrefcount(n) for n in nodes] == before
