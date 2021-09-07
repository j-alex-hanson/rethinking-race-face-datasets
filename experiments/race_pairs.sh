#!/bin/bash

USER="default"
EXPERIMENT="race_pairs"
METHOD="arcface"
TRAIN_RACE="African"
TRAIN_RACE_B="Asian"
PERCENTS=(0 25 50 75 100)
RACES=($TRAIN_RACE $TRAIN_RACE_B)
NUM_CLASSES=5000

for i in `seq 0 4`; do
    for j in ${PERCENTS[@]}; do
        python main.py --usr-config $USER --method $METHOD --train-race $TRAIN_RACE --train-race-b $TRAIN_RACE_B --percent-primary $j --experiment-name $EXPERIMENT --trial-num $i --num-classes $NUM_CLASSES --train

        python main.py --usr-config $USER --test-race $TRAIN_RACE --method $METHOD --train-race $TRAIN_RACE --train-race-b $TRAIN_RACE_B --percent-primary $j --experiment-name $EXPERIMENT --trial-num $i --num-classes $NUM_CLASSES --test
        python main.py --usr-config $USER --test-race $TRAIN_RACE_B --method $METHOD --train-race $TRAIN_RACE --train-race-b $TRAIN_RACE_B --percent-primary $j --experiment-name $EXPERIMENT --trial-num $i --num-classes $NUM_CLASSES --test
   done
done
