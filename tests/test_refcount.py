import ast
import gc
import sys
import weakref
from collections import Counter

import pytest

from fast_walk import walk_dfs as fast_walk


SOURCE = """
def foo(x, y):
    a = 1 + 2
    b = [x, y, a]
    for i in b:
        if i > 0:
            print(i, "positive")
        else:
            yield i

class C:
    def m(self):
        return [1, 2, 3]
"""


def test_result_matches_ast_walk():
    tree = ast.parse(SOURCE)
    fast_ids = [id(n) for n in fast_walk(tree)]
    ast_ids = [id(n) for n in ast.walk(tree)]
    assert sorted(fast_ids) == sorted(ast_ids)


def test_refcount_neutral_after_walks():
    """Calling walk N times then dropping the result must leave input
    refcounts exactly where they started. Drift means we're either
    over-INCREF'ing (leak) or under-INCREF'ing (use-after-free waiting to
    happen)."""
    tree = ast.parse(SOURCE)
    sample = list(ast.walk(tree))

    # Drain any prior-test garbage (shared AST singletons like ast.Load()
    # have refcounts that reflect every live AST tree in the process,
    # so leftover trees from earlier tests would inflate `before`).
    gc.collect()
    before = [sys.getrefcount(n) for n in sample]
    for _ in range(1000):
        fast_walk(tree)  # result is dropped immediately as an expr statement
    gc.collect()
    after = [sys.getrefcount(n) for n in sample]

    drifts = [
        (i, b, a) for i, (b, a) in enumerate(zip(before, after)) if b != a
    ]
    assert not drifts, f"refcount drift on {len(drifts)} nodes: {drifts[:5]}"


def test_result_holds_strong_refs():
    """Items in the returned list should stay alive as long as the list
    does, even if every other reference to them is dropped. This catches
    a missing Py_INCREF — without it, items would be freed when the last
    *non-list* reference is dropped, and the list would hold dangling
    pointers."""
    tree = ast.parse(SOURCE)
    result = fast_walk(tree)
    # weakrefs let us observe liveness without contributing to refcount
    weak = []
    for n in result:
        try:
            weak.append(weakref.ref(n))
        except TypeError:
            pass
    assert weak, "test setup: expected at least one weakref-able node"

    del tree
    gc.collect()
    alive = sum(1 for w in weak if w() is not None)
    assert alive == len(weak), (
        f"{len(weak) - alive} nodes freed while still held by result list"
    )


def test_refcount_of_result_items():
    """Each node's refcount bump must equal its appearance count in the
    result. Some AST singletons (ast.Load(), ast.Store(), ast.Del(), ...)
    are shared across many parents, so they legitimately appear multiple
    times in a walk — this test uses ast.walk as the spec for expected
    appearance counts and checks fast_walk's refcount deltas match."""
    tree = ast.parse(SOURCE)
    sample = list(ast.walk(tree))
    expected_counts = Counter(id(n) for n in ast.walk(tree))

    before = [sys.getrefcount(n) for n in sample]
    result = fast_walk(tree)
    after = [sys.getrefcount(n) for n in sample]

    mismatches = [
        (i, b + expected_counts[id(n)], a)
        for i, (n, b, a) in enumerate(zip(sample, before, after))
        if a != b + expected_counts[id(n)]
    ]
    assert not mismatches, (
        f"refcount delta != appearance count for {len(mismatches)} nodes: "
        f"{mismatches[:5]}"
    )

    del result  # silence unused-var lint + explicit teardown


def test_survives_gc_between_walks():
    """Stress test: forcing full GC between walks should not crash or
    invalidate the returned list."""
    tree = ast.parse(SOURCE)
    for _ in range(100):
        result = fast_walk(tree)
        gc.collect()
        # touch every item to force a read — segfaults surface here
        assert sum(1 for _ in result) == len(result)


def test_empty_input():
    tree = ast.parse("")
    assert [id(n) for n in fast_walk(tree)] == [id(n) for n in ast.walk(tree)]


@pytest.mark.parametrize("iterations", [10, 100, 1000])
def test_no_leak_growth(iterations):
    """Run many walks of different sizes. If we leak on every walk,
    memory would grow unbounded — we can't measure RSS portably, but we
    can at least detect that the result list's refcount bookkeeping is
    stable regardless of iteration count."""
    tree = ast.parse(SOURCE * 5)
    sample = list(ast.walk(tree))[:10]
    before = [sys.getrefcount(n) for n in sample]
    for _ in range(iterations):
        fast_walk(tree)
    gc.collect()
    after = [sys.getrefcount(n) for n in sample]
    assert before == after
