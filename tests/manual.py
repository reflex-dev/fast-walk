if __name__ == "__main__":
    from ast import parse
    from pathlib import Path
    from fast_walk import walk_unordered
    import difflib

    source_code = Path(difflib.__file__).read_text()
    node = parse(source_code)

    def walk_benchmark():
        walk_unordered(node)

    for _ in range(1_000):
        walk_benchmark()
