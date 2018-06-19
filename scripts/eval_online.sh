#!/usr/bin/env bash

cwd=$(pwd)

if [ $# == 0 ]; then
    echo "Supply rec electrode folder (includeing final /), mode (block - ff - reg)"
elif [ $# == 2 ]; then
    folder=$1
    analysis=$2
    recordings=$(ls $folder)

    if [ $analysis == 'dimred' ]; then
        dim='5 10 20 30 50 75 100'

        echo 'Dimensionality reduction analysis'
        for r in $recordings
        do
            for m in $dim
            do
                echo $m $r
                python ../online_analysis.py -r $folder$r -M $m
            done
        done
    fi
fi
