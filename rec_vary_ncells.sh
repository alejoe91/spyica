#!/usr/bin/env bash

if [ $# == 0 ];
  then
    echo "Supply spike path, ncells ('list'), duration, noise, seed"
else
    if [ $# == 1 ] && [ "$1" == "all" ];
      then
        echo "Spikesorting all recordings"
    else
        spikes=$1
        ncells=$2
        dur=$3
        noise=$4
        seed=$5
        for n in $ncells
        do
        echo "Generating recording: ncells $n duration $dur noise $noise"
        python generate_recordings.py -f $spikes -dur $dur -ncells $n -noiselev $noise -sync 0 -noplot -noisemod
        done
        echo All done
    fi
fi