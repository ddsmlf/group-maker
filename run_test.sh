#!/bin/bash

export PYTHONPATH=$(pwd):$(pwd)/tests

for test_file in tests/*_test.py; do
    test_module=$(basename "$test_file" .py)
    echo "Running test $test_module"
    python3 -m unittest "tests.$test_module"
    if [ $? -ne 0 ]; then
        echo "Test $test_module failed"
        exit 1
    fi
done

echo "All tests passed"