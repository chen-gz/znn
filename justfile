# Download and extract Fashion MNIST dataset files
download-data:
    bash scripts/download_data.sh

# Run all unit and integration tests
test:
    zig build test

# Run the MLP Fashion MNIST training pipeline
run:
    zig build run


# Run the Linear Regression optimization demo
run-lr:
    zig build run-lr

# Run the CNN Fashion MNIST training pipeline
run-cnn:
    zig build run-cnn

# Run the performance benchmark suite (supports e.g. just bench --filter gemm)
bench *args:
    zig build run-bench -Doptimize=ReleaseFast -- {{args}}

# Generate code coverage report using kcov and print summary table
coverage:
    bash scripts/coverage.sh

# Generate code coverage report and open in default browser
coverage-open:
    bash scripts/coverage.sh --open

