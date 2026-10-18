# MLX-C Subproject & Prebuilt Acceleration Libraries for Apple Silicon

This directory contains the vendored [Apple MLX C API (`mlx-c`)](https://github.com/ml-explore/mlx-c) subproject and prebuilt Apple Silicon libraries:

- `include/`: Official C API headers (`mlx/c/*.h`).
- `lib/`: Prebuilt Apple Silicon libraries (`libmlxc.dylib`, `libmlx.dylib`, `libjaccl.dylib`) with relative `@rpath` configuration, requiring zero Homebrew dependencies on macOS.
- `src/`: Complete source code of `mlx-c` for reference and subproject build tracking.

## Licenses
- `mlx-c`: MIT License (Copyright © 2023-2024 Apple Inc.)
- `mlx`: MIT & Apache 2.0 License (Copyright © 2023-2024 Apple Inc.)
