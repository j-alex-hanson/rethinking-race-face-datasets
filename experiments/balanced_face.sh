#!/bin/bash

USER="default"
EXPERIMENT="balanced_face"
METHOD="arcface"
RACES=("African" "Asian" "Caucasian" "Indian")
NUM_CLASSES=28000

for i in `seq 0 4`; do
   python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --num-classes $NUM_CLASSES --trial-num $i --train
    
    for j in ${RACES[@]}; do
        python main.py --usr-config $USER --test-race $j --method $METHOD --experiment-name $EXPERIMENT --num-classes $NUM_CLASSES --trial-num $i --test
    done
done
