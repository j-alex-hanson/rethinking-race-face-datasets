#!/bin/bash

USER="default"
EXPERIMENT="race_simplex"
METHOD="arcface"
RACES=("African" "Asian" "Caucasian" "Indian")
NUM_CLASSES=5000
SIMPLEX_MAXES=(30 40 60 100)
SIMPLEX_POINT_IDS=`seq 0 0`

for i in ${SIMPLEX_POINT_IDS[@]}; do
    for j in ${SIMPLEX_MAXES[@]}; do
        for k in `seq 0 0`; do
            python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --simplex-max $j --simplex-point-id $i --num-classes $NUM_CLASSES --trial-num $k --train
            
            for l in ${RACES[@]}; do
                python main.py --usr-config $USER --test-race $l --method $METHOD --experiment-name $EXPERIMENT --simplex-max $j --simplex-point-id $i --num-classes $NUM_CLASSES --trial-num $k --test
            done
        done
    done
done

for k in `seq 0 0`; do
    python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --simplex-max 25 --simplex-point-id 0 --num-classes $NUM_CLASSES --trial-num $k --train
    
    for l in ${RACES[@]}; do
        python main.py --usr-config $USER --test-race $l --method $METHOD --experiment-name $EXPERIMENT --simplex-max 25 --simplex-point-id 0 --num-classes $NUM_CLASSES --trial-num $k --test
    done
done
