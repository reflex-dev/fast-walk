from ast import AST, NodeVisitor as ASTNodeVisitor, parse
from ast import walk as ast_walk
import ast
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


class NoStrRuleVisitor(ASTNodeVisitor):
    def __init__(self):
        self.violations: list[str] = []
        self.inside_class = False

    def visit_ClassDef(self, node: ast.ClassDef):
        previous_inside_class = self.inside_class
        self.inside_class = True
        node = self.generic_visit(node)
        self.inside_class = previous_inside_class
        return node

    def visit_Call(self, node: ast.Call):
        if self.inside_class:
            return self.generic_visit(node)

        for arg in node.args:
            if (
                not isinstance(arg, ast.Call)
                or not isinstance(arg.func, ast.Name)
                or arg.func.id != "str"
            ):
                continue

            self.violations.append(
                "Do not use `str` outside of classes.",
            )

        return self.generic_visit(node)


def ast_visitor_visit(tree: ast.Module):
    visitor = NoStrRuleVisitor()
    visitor.visit(tree)
    return visitor.violations


all_ast_nodes = {
    cls
    for cls in ast.__dict__.values()
    if isinstance(cls, type) and issubclass(cls, AST)
}


class PythonVisitor:
    def __init_subclass__(cls) -> None:
        visitors = {
            ast_class: visitor
            for name, visitor in cls.__dict__.items()
            if name.startswith("visit_")
            and (ast_class_name := name.removeprefix("visit_")) in ast.__dict__
            and isinstance((ast_class := getattr(ast, ast_class_name)), type)
            and issubclass(ast_class, AST)
        }
        cls._visitors = visitors
        super().__init_subclass__()

    def visit(self, node):
        """Visit a node."""
        if (visitor := self._visitors.get(type(node), None)) is not None:
            return visitor(self, node)
        return self.generic_visit(node)

    def generic_visit(self, node: AST):
        """Called if no explicit visitor function exists for a node."""
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, AST):
                if (visitor := self._visitors.get(type(value), None)) is not None:
                    visitor(self, value)
                    continue
                self.generic_visit(value)
            elif type(value) is list:
                for item in value:
                    if isinstance(item, AST):
                        if (
                            visitor := self._visitors.get(type(item), None)
                        ) is not None:
                            visitor(self, item)
                            continue
                        self.generic_visit(item)

    def visit_Constant(self, node):
        value = node.value
        type_name = ast._const_node_type_names.get(type(value))
        if type_name is None:
            for cls, name in ast._const_node_type_names.items():
                if isinstance(value, cls):
                    type_name = name
                    break
        if type_name is not None:
            method = "visit_" + type_name
            try:
                visitor = getattr(self, method)
            except AttributeError:
                pass
            else:
                import warnings

                warnings.warn(
                    f"{method} is deprecated; add visit_Constant", DeprecationWarning, 2
                )
                return visitor(node)
        return self.generic_visit(node)


class NoStrRulePythonVisitor(PythonVisitor):
    def __init__(self):
        self.violations: list[str] = []
        self.inside_class = False

    def visit_ClassDef(self, node: ast.ClassDef):
        previous_inside_class = self.inside_class
        self.inside_class = True
        node = self.generic_visit(node)
        self.inside_class = previous_inside_class
        return node

    def visit_Call(self, node: ast.Call):
        if self.inside_class:
            return self.generic_visit(node)

        for arg in node.args:
            if (
                not isinstance(arg, ast.Call)
                or not isinstance(arg.func, ast.Name)
                or arg.func.id != "str"
            ):
                continue

            self.violations.append(
                "Do not use `str` outside of classes.",
            )

        return self.generic_visit(node)


def python_visitor_visit(node: ast.Module):
    visitor = NoStrRulePythonVisitor()
    visitor.visit(node)
    return visitor.violations


@pytest.mark.parametrize(
    "algorithm",
    [ast_visitor_visit, python_visitor_visit],
)
def test_visit(benchmark: BenchmarkFixture, algorithm: Callable[[AST], list[str]]):
    import difflib

    source_code = Path(difflib.__file__).read_text()
    node = parse(source_code)

    def run():
        for _ in algorithm(node):
            pass

    benchmark(run)
