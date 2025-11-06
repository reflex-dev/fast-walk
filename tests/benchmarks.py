from ast import AST, parse
from ast import walk as ast_walk
from collections.abc import Callable, Iterable
from pathlib import Path

from pytest_codspeed import BenchmarkFixture
from fast_walk import walk as fast_walk
import pytest


def python_walk_helper(node: AST, fields: tuple[str, ...], nodes: list[AST]):
    nodes.append(node)
    for field in fields:
        value = getattr(node, field, None)
        if type(value) is list:
            for item in value:
                if subfields := getattr(item, "_fields", None):
                    python_walk_helper(item, subfields, nodes)
        elif subfields := getattr(value, "_fields", None):
            python_walk_helper(value, subfields, nodes)  # pyright: ignore[reportArgumentType]


def python_walk(node: AST) -> list[AST]:
    nodes: list[AST] = []
    python_walk_helper(node, node._fields, nodes)
    return nodes


@pytest.mark.parametrize(
    "algorithm",
    [
        ast_walk,
        fast_walk,
        python_walk,
    ],
)
def test_walk(benchmark: BenchmarkFixture, algorithm: Callable[[AST], Iterable[AST]]):
    import difflib

    source_code = Path(difflib.__file__).read_text()
    node = parse(source_code)

    def run():
        for _ in algorithm(node):
            pass

    benchmark(run)
