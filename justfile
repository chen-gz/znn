# Run all unit and integration tests
test:
    zig build test

# Run the MLP Fashion MNIST training pipeline
run:
    zig build run

# Run the Linear Regression optimization demo
run-lr:
    zig build run-lr

# Generate code coverage report using kcov
coverage:
    #!/usr/bin/env bash
    if ! command -v kcov &> /dev/null; then
        echo "Error: kcov is not installed on your system."
        if [ "$(uname)" = "Darwin" ]; then
            echo "Please install kcov via Homebrew: 'brew install kcov'"
        else
            echo "Please install kcov using your system package manager."
        fi
        exit 1
    fi
    echo "Building tests and installing test binaries..."
    zig build test --summary none
    echo "Running kcov..."
    rm -rf ./coverage-report
    if [ -f "zig-out/bin/root_tests" ]; then
        echo "Profiling root_tests..."
        kcov --include-path=./src ./coverage-report zig-out/bin/root_tests
    fi
    if [ -f "zig-out/bin/exe_tests" ]; then
        echo "Profiling exe_tests..."
        kcov --include-path=./src ./coverage-report zig-out/bin/exe_tests
    fi
    echo "Coverage report generated at ./coverage-report/index.html"
