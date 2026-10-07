"""The functions here read each
node's children by `_fields` name, so they must agree with the stdlib on
parsed trees and on nodes a transformer built in another attribute order."""

from __future__ import annotations

import ast
import copy
import gc
import sys
import textwrap

import pytest

from fast_walk import (
    fix_missing_locations,
    set_parents,
    subtree_hash,
    subtree_hashes,
    walk_frontier_events,
    walk_of_types,
)

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


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def pruned_pre_order(node: ast.AST, prune: tuple[type, ...], root: bool = True):
    yield node
    if root or type(node) not in prune:
        for child in ast.iter_child_nodes(node):
            yield from pruned_pre_order(child, prune, root=False)


@pytest.mark.parametrize("kinds", [(ast.Name,), (ast.Return, *SCOPES), ()])
def test_walk_of_types_stops_below_a_pruned_node(kinds):
    tree = ast.parse(SOURCE)
    for root in [tree, *walk_of_types(tree, SCOPES)]:
        assert [id(n) for n in walk_of_types(root, kinds, SCOPES)] == [
            id(n) for n in pruned_pre_order(root, SCOPES) if type(n) in kinds
        ]


def test_set_parents_matches_a_pre_order_loop():
    ours, theirs = ast.parse(SOURCE), ast.parse(SOURCE)
    set_parents(ours)
    for node in pre_order(theirs):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    assert not hasattr(ours, "parent")
    position = {id(n): i for i, n in enumerate(pre_order(theirs))}
    position |= {id(n): i for i, n in enumerate(pre_order(ours))}
    assert [
        position[id(n.parent)] for n in pre_order(ours) if hasattr(n, "parent")
    ] == [position[id(n.parent)] for n in pre_order(theirs) if hasattr(n, "parent")]


def test_set_parents_gives_a_shared_node_its_last_parent():
    shared = ast.Name(id="x", ctx=ast.Load())
    tree = ast.parse("f(a)\ng(b)")
    tree.body[0].value.args[0] = shared
    tree.body[1].value.args[0] = shared
    set_parents(tree)
    assert shared.parent is tree.body[1].value


def reference_events(node, kinds, bracket):
    events = []

    def descend(current):
        if type(current) in bracket:
            events.append((1, current))
        for child in ast.iter_child_nodes(current):
            if type(child) in kinds:
                events.append((0, child))
            else:
                descend(child)
        if type(current) in bracket:
            events.append((2, current))

    descend(node)
    return events


EXPRESSIONS = tuple(
    kind for kind in vars(ast).values()
    if isinstance(kind, type) and issubclass(kind, ast.expr) and kind is not ast.expr
)


@pytest.mark.parametrize(
    ("kinds", "bracket"),
    [
        ((ast.Name, ast.Assign), EXPRESSIONS),
        ((ast.Call,), (ast.Module, ast.Attribute)),
        ((), EXPRESSIONS),
        ((ast.FunctionDef,), ()),
    ],
)
def test_walk_frontier_events_matches_a_recursive_descent(kinds, bracket):
    tree = ast.parse(SOURCE)
    for root in [tree, *walk_of_types(tree, (ast.Call, ast.Assign))]:
        expected = reference_events(root, kinds, bracket)
        assert [(e, id(n)) for e, n in walk_frontier_events(root, kinds, bracket)] == [
            (e, id(n)) for e, n in expected
        ]


def test_walk_frontier_events_on_a_deep_chain():
    chain: ast.expr = ast.Name(id="a", ctx=ast.Load())
    for _ in range(50_000):
        chain = ast.Attribute(value=chain, attr="b", ctx=ast.Load())
    events = walk_frontier_events(chain, (ast.Name,), EXPRESSIONS)
    assert len(events) == 2 * 50_000 + 1


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


@pytest.mark.parametrize(
    ("built", "source"),
    [
        (lambda: ast.Constant(value="x"), '"x"'),
        (lambda: ast.Constant(value="x", kind=None), '"x"'),
        (lambda: ast.Call(func=ast.Name(id="f"), args=[], keywords=[]), "f()"),
    ],
)
def test_a_field_a_constructor_left_out_hashes_as_its_default(built, source):
    node, parsed = built(), ast.parse(source).body[0].value
    assert ast.dump(node) == ast.dump(parsed)
    assert subtree_hash(node) == subtree_hash(parsed)


def test_a_field_a_constructor_left_out_is_not_a_child():
    node = ast.Constant(value="x")
    assert walk_of_types(node, (ast.Constant,)) == [node]
    assert walk_frontier_events(node, (), (ast.Constant,)) == [(1, node), (2, node)]


def test_subtree_hash_is_the_roots_entry():
    tree = ast.parse(SOURCE)
    assert subtree_hash(tree) == subtree_hashes(tree)[id(tree)]


def test_subtree_hashes_ignore_dict_order():
    built = ast.Call(lineno=1, keywords=[], func=ast.Name(id="f", ctx=ast.Load()), args=[])
    parsed = ast.parse("f()").body[0].value
    assert subtree_hashes(built)[id(built)] == subtree_hashes(parsed)[id(parsed)]


def test_refcount_neutral():
    tree = ast.parse(SOURCE)
    set_parents(tree)
    nodes = list(ast.walk(tree))
    gc.collect()
    before = [sys.getrefcount(n) for n in nodes]
    for _ in range(50):
        walk_of_types(tree, (ast.Name, ast.Call))
        subtree_hashes(tree)
        subtree_hash(tree)
        set_parents(tree)
        fix_missing_locations(tree)
        walk_of_types(tree, (ast.Name,), SCOPES)
        walk_frontier_events(tree, (ast.Name,), EXPRESSIONS)
    gc.collect()
    assert [sys.getrefcount(n) for n in nodes] == before
