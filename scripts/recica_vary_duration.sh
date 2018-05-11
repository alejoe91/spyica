#!/usr/bin/env bash

if [ $# == 0 ];
  then
    echo "Supply spike path, ncells, duration ('list'), noise, nrec"
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
        for n in $dur
        do
        echo "Generating recording: ncells $ncells duration $n noise $noise"
        for i in `seq 1 $nrec`;
            do
                echo $i $n
                python ../generate_gt_ica_recordings.py -f $spikes -dur $n -ncells $ncells -noiselev $noise -sync 0 -noplot -elmod
            done
        done
        echo All done
    fi
fi