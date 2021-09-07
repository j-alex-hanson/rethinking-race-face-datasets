#!/bin/bash

USER="default"
EXPERIMENT="single_race"
METHOD="arcface"
TRAIN_RACE="African"
TEST_RACES=("African" "Asian" "Caucasian" "Indian")

for i in `seq 0.000 0.001 0.010`; do
    echo \"Noise fraction $i\"
    python main.py --method $METHOD --usr-config $USER --train-race $TRAIN_RACE --data-noise $i --experiment-name $EXPERIMENT --train
    
    for j in ${TEST_RACES[@]}; do
        python main.py --test-race $j --method $METHOD --usr-config $USER --train-race $TRAIN_RACE --data-noise $i --experiment-name $EXPERIMENT --test
    done
done
