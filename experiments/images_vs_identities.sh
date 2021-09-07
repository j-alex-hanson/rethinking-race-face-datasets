#!/bin/bash

USER="default"
EXPERIMENT="images_vs_identities"
METHOD="arcface"
RACES=("African" "Asian" "Caucasian" "Indian")

for i in `seq 0 4`; do
    python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --num-classes 10000 --experiment-race African --num-extra-images 0 --trial-num $i --train
    
    for j in ${RACES[@]}; do
        python main.py --usr-config $USER --test-race $j --method $METHOD --experiment-name $EXPERIMENT --num-classes 10000 --experiment-race African --num-extra-images 0 --trial-num $i --test
    done
done

for i in ${RACES[@]}; do
    for j in `seq 0 4`; do
        python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --num-classes 11250 --experiment-race $i --num-extra-images 0 --trial-num $j --train
        
        for k in ${RACES[@]}; do
            python main.py --usr-config $USER --test-race $k --method $METHOD --experiment-name $EXPERIMENT --num-classes 11250 --experiment-race $i --num-extra-images 0 --trial-num $j --test
        done
        
        python main.py --usr-config $USER --method $METHOD --experiment-name $EXPERIMENT --num-classes 10000 --experiment-race $i --num-extra-images 5 --trial-num $j --train
        
        for k in ${RACES[@]}; do
            python main.py --usr-config $USER --test-race $k --method $METHOD --experiment-name $EXPERIMENT --num-classes 10000 --experiment-race $i --num-extra-images 5 --trial-num $j --test
        done
    done
done
