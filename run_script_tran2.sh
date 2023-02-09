#!/bin/bash

epsilons=( 0.01 1 100 1000 -0.01 -1 -100 -1000 )

for eps in "${epsilons[@]}"
do
    for index in {1234..1243}
    do
        nohup python3 run_main_lin_$1.py tran2 $eps $index &
        sleep .5
    done
done