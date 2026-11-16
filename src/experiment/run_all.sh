#!/bin/bash
# Run all 6 ablation experiments sequentially.
# Usage: bash experiment/run_all.sh
# Make sure to run this from src/ directory.

set -e

cd "$(dirname "$0")/.."  # navigate to src/

echo "========================================"
echo " Starting ablation experiments (6 total) "
echo "========================================"
echo ""

for i in 1 2 3 4 5 6 7 8 9 10; do
    echo ""
    echo ">>>> Running experiment $i/6 <<<<"
    python experiment/run_one.py --id "$i"
    echo ""
done

echo "========================================"
echo " All experiments complete!"
echo "========================================"
python experiment/show_table.py
