#!/usr/bin/env bash

# Run test.py 50 times
for i in {1..100}; do
  echo "Iteration $i"
  python test.py ./restaurant-data.txt
done
