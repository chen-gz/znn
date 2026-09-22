#!/usr/bin/env bash
set -e

# ==============================================================================
# Automated Code Coverage Script for ZNN (Zig Neural Network)
# ==============================================================================
# Usage:
#   ./scripts/coverage.sh [OPTIONS]
#
# Options:
#   --open            Open HTML coverage report in browser upon completion
#   --fail-under=N    Exit with error code 1 if total coverage is below N%
#   --clean           Clean coverage-report directory before running
#   --help            Show this help message
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/coverage-report"
FAIL_UNDER=0
OPEN_REPORT=false
CLEAN_DIR=false

for arg in "$@"; do
    case "$arg" in
        --open)
            OPEN_REPORT=true
            ;;
        --fail-under=*)
            FAIL_UNDER="${arg#*=}"
            ;;
        --clean)
            CLEAN_DIR=true
            ;;
        --help|-h)
            echo "Usage: ./scripts/coverage.sh [--open] [--fail-under=N] [--clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./scripts/coverage.sh [--open] [--fail-under=N] [--clean]"
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

# 1. Check for kcov installation
if ! command -v kcov &> /dev/null; then
    echo -e "\033[1;31m[ERROR]\033[0m kcov is not installed on your system."
    if [ "$(uname)" = "Darwin" ]; then
        echo -e "Install via Homebrew: \033[1;32mbrew install kcov\033[0m"
    elif command -v apt-get &> /dev/null; then
        echo -e "Install via apt: \033[1;32msudo apt-get install kcov\033[0m"
    elif command -v pacman &> /dev/null; then
        echo -e "Install via pacman: \033[1;32msudo pacman -S kcov\033[0m"
    else
        echo "Please install kcov using your system's package manager."
    fi
    exit 1
fi

# 2. Clean if requested
if [ "$CLEAN_DIR" = true ]; then
    echo "Cleaning ${REPORT_DIR}..."
    rm -rf "${REPORT_DIR}"
fi

# 3. Build test suite and install test binaries to zig-out/bin
echo -e "\033[1;34m[1/3]\033[0m Building test suite and installing binaries..."
zig build test --summary none

# Ensure test binaries exist
BINARIES=()
for b in "root_tests" "cnn_tests" "exe_tests"; do
    if [ -f "zig-out/bin/${b}" ]; then
        BINARIES+=("zig-out/bin/${b}")
    fi
done

if [ ${#BINARIES[@]} -eq 0 ]; then
    echo -e "\033[1;31m[ERROR]\033[0m No test binaries found in zig-out/bin/."
    exit 1
fi

# 4. Run kcov on test binaries
echo -e "\033[1;34m[2/3]\033[0m Profiling execution with kcov..."
for bin in "${BINARIES[@]}"; do
    bin_name="$(basename "${bin}")"
    echo "  -> Profiling ${bin_name}..."
    kcov --include-path=./src "${REPORT_DIR}" "${bin}" > /dev/null 2>&1
done

# 5. Extract and print summary table
echo -e "\033[1;34m[3/3]\033[0m Generating coverage summary..."

COVERAGE_JSON=""
if [ -f "${REPORT_DIR}/kcov-merged/coverage.json" ]; then
    COVERAGE_JSON="${REPORT_DIR}/kcov-merged/coverage.json"
elif [ -f "${REPORT_DIR}/root_tests/coverage.json" ]; then
    COVERAGE_JSON="${REPORT_DIR}/root_tests/coverage.json"
else
    # Find any coverage.json
    COVERAGE_JSON="$(find "${REPORT_DIR}" -name "coverage.json" | head -n 1)"
fi

if [ -z "${COVERAGE_JSON}" ] || [ ! -f "${COVERAGE_JSON}" ]; then
    echo -e "\033[1;31m[ERROR]\033[0m Could not locate coverage.json in ${REPORT_DIR}"
    exit 1
fi

python3 - <<EOF
import json
import os
import sys

json_path = "${COVERAGE_JSON}"
with open(json_path, 'r') as f:
    data = json.load(f)

files = data.get("files", [])
total_percent = float(data.get("percent_covered", 0.0))
covered_lines = int(data.get("covered_lines", 0))
total_lines = int(data.get("total_lines", 0))

# ANSI Color codes
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

def color_percent(p):
    if p >= 90.0:
        return f"{GREEN}{p:6.2f}%{RESET}"
    elif p >= 80.0:
        return f"{YELLOW}{p:6.2f}%{RESET}"
    else:
        return f"{RED}{p:6.2f}%{RESET}"

print("\n" + "=" * 78)
print(f"{BOLD}{'File / Module':<45} {'Covered':>9} {'Total':>7} {'Coverage':>10}{RESET}")
print("-" * 78)

# Sort files alphabetically or by percent
base_dir = os.path.abspath("${PROJECT_ROOT}") + "/"
files.sort(key=lambda x: x.get("file", ""))

for item in files:
    fpath = item.get("file", "")
    rel_path = fpath.replace(base_dir, "")
    pct = float(item.get("percent_covered", 0.0))
    cov = int(item.get("covered_lines", 0))
    tot = int(item.get("total_lines", 0))
    print(f"{rel_path:<45} {cov:>9} {tot:>7} {color_percent(pct):>10}")

print("-" * 78)
status_color = GREEN if total_percent >= 90.0 else (YELLOW if total_percent >= 80.0 else RED)
print(f"{BOLD}{'TOTAL':<45} {covered_lines:>9} {total_lines:>7} {status_color}{BOLD}{total_percent:6.2f}%{RESET}")
print("=" * 78)

report_html = os.path.abspath("${REPORT_DIR}/index.html")
print(f"\n{BOLD}Interactive HTML Report:{RESET} {CYAN}file://{report_html}{RESET}\n")

fail_under = float("${FAIL_UNDER}")
if fail_under > 0 and total_percent < fail_under:
    print(f"{RED}{BOLD}[FAILED]{RESET} Total coverage {total_percent:.2f}% is below required threshold of {fail_under:.2f}%")
    sys.exit(1)
EOF

STATUS=$?
if [ $STATUS -ne 0 ]; then
    exit $STATUS
fi

# 6. Optionally open HTML report
if [ "$OPEN_REPORT" = true ]; then
    echo "Opening coverage report in default browser..."
    if [ "$(uname)" = "Darwin" ]; then
        open "${REPORT_DIR}/index.html"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "${REPORT_DIR}/index.html"
    fi
fi
