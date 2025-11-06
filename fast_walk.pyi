import ast

def walk(node: ast.AST) -> list[ast.AST]:
    """Return a list of all AST nodes in the tree rooted at `node`."""
