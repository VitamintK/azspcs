#!/bin/bash
for i in $(seq $2)
do
    python3 main.py --n $1 --its $3
done