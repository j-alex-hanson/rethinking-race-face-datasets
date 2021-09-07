#!/bin/bash

USER="default"
EXPERIMENT="single_race"
METHOD="arcface"
TRAIN_RACE="African"
TEST_RACES=("African" "Asian" "Caucasian" "Indian")

for i in `seq 0 4`; do
    python main.py --usr-config $USER --method $METHOD --train-race $TRAIN_RACE --experiment-name $EXPERIMENT --trial-num $i --train
    
    for j in ${TEST_RACES[@]}; do
        python main.py --usr-config $USER --test-race $j --method $METHOD --train-race $TRAIN_RACE --experiment-name $EXPERIMENT --trial-num $i --test
    done
done
