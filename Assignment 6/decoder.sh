#!/bin/sh

if [ -z "$3" ]; then
    python3 src/task2.py --decoder --grid_file $1 --value_and_policy_file $2
else
    python3 src/task2.py --decoder --grid_file $1 --value_and_policy_file $2 --probability $3
fi