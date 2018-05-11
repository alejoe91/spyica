#!/usr/bin/env bash

if [ $# == 0 ];
  then
    echo "Supply spike path, ncells, duration, noise ('list'), nrec"
else
    if [ $# == 1 ] && [ "$1" == "all" ];
      then
        echo "Spikesorting all recordings"
    else
        spikes=$1
        ncells=$2
        dur=$3
        noise=$4
        nrec=$5
        for n in $noise
        do
        echo "Generating recording: ncells $n duration $dur noise $noise"
        for i in `seq 1 $nrec`;
            do
                echo $i
                python ../generate_gt_ica_recordings.py -f $spikes -dur $dur -ncells $ncells -noiselev $n -sync 0 -noplot -elmod
            done
        done
        echo All done
    fi
fi