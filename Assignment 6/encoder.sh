#!/bin/sh

if [ -z "$2" ]; then
    python3 src/task2.py --encoder --grid_file $1
else
    python3 src/task2.py --encoder --grid_file $1 --probability $2
fi