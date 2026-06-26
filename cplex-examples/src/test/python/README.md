# CPLEX Examples Test Suite

This directory contains pytest tests that ensure all examples in `src/main/python` execute without throwing exceptions.

## Overview

The test suite (`test_examples.py`) validates that each CPLEX example runs successfully as a standalone script with appropriate command-line arguments and data files. Each example is executed as a subprocess, mimicking how users would run them from the command line.

## Installation

First, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Or install pytest separately:

```bash
pip install pytest
```

## Running the Tests

### Run all tests:
```bash
pytest src/test/python/test_examples.py -v
```

### Run a specific test:
```bash
pytest src/test/python/test_examples.py::TestExamples::test_lpex1_row -v
```

### Run tests matching a pattern:
```bash
pytest src/test/python/test_examples.py -k "lpex" -v
```

### Run with output capture disabled (to see example output):
```bash
pytest src/test/python/test_examples.py -v -s
```

## Test Coverage

The test suite covers all examples including:

- **LP Examples**: lpex1-lpex7 (linear programming)
- **MIP Examples**: mipex1-mipex4 (mixed integer programming)
- **QP Examples**: qpex1-qpex2, qcpex1, miqpex1 (quadratic programming)
- **Advanced MIP**: admipex1-admipex9 (callbacks, cuts, branching)
- **Benders Decomposition**: benders, bendersatsp, bendersatsp2
- **Application Examples**: diet, facility, cutstock, etsp, steel, warehouse, etc.
- **Special Features**: conflictex1, populate, multiobjex1, genericbranch

## Test Structure

Each test:
1. Runs the example script as a subprocess (mimicking command-line execution)
2. Passes appropriate command-line arguments
3. Uses data files from `src/main/data` when required
4. Validates that the script exits with status code 0 (no exceptions)
5. Captures stdout/stderr for debugging if the script fails

## Notes

- Some examples require specific data files (e.g., `.mps`, `.lp`, `.dat` files)
- Examples with multiple execution modes are tested with different parameter combinations
- The tests use relative paths to locate data files automatically
- CPLEX must be properly installed and licensed for the tests to run

## Troubleshooting

If tests fail:
1. Ensure CPLEX is properly installed: `python -c "import cplex; print(cplex.__version__)"`
2. Check that all data files exist in `src/main/data`
3. Verify CPLEX license is valid
4. Run individual tests with `-s` flag to see detailed output

## Adding New Tests

When adding a new example to `src/main/python`:
1. Add a corresponding test method to the `TestExamples` class
2. Provide appropriate arguments and data files
3. Follow the naming convention: `test_<example_name>`
4. Document any special requirements in the test docstring