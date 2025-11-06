# fast-walk

A fast (almost) drop-in implementation of `ast.walk`.

## Installation

```bash
pip install fast-walk
```

## Usage

```python
from fast_walk import walk
import parse

code = """
def hello(  x,y,   z  ):
    print( x+y+z )
"""

for node in walk(ast.parse(code)):
    pass
```

## Development

### Prerequisites

- Rust (latest stable)
- Python 3.13+
- [maturin](https://github.com/PyO3/maturin)

### Building from source

```bash
# Install maturin
pip install maturin

# Build the package
maturin develop

# Or build a release version
maturin build --release
```

## License

MIT

## Links

- [Repository](https://github.com/reflex-dev/fast-walk)
