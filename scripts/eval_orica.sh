#!/usr/bin/env bash

cwd=$(pwd)
ff=cooling
oricamod='A_block W_block original'
chdir=$PWD

if [ $# == 0 ]; then
    echo "Supply rec electrode folder (includeing final /), mode (block - ff - reg - ica - remove)"
elif [ $# == 2 ]; then
    folder=$1
    analysis=$2
    recordings=$(ls $folder)

    if [ $analysis == 'block' ]; then
        #blocks='1 5 10 20 35 50 75 100 200 350 500 750 1000 1500 2000'
        blocks='1 5 10 20  50  100  500 1000  2000'

        echo 'Block analysis'


        for r in $recordings
        do
            for mod in $oricamod
            do
                for bl in $blocks
                do
                    echo $mod $bl
                    python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -noplot
                done
            done
        done

    elif [ $analysis == 'ff' ]; then
        echo 'FF analysis'
        ff='cooling constant'
        #lambda='N 5 3 0.995 0.75 0.5 0.25 0.1 0.01 0.001 0.0001 0.00001'
        lambda='N 5  3 0.995 0.5  0.1 0.005 0.00001'
        bl=50

        for r in $recordings
        do
            for mod in $oricamod
            do
                for f in $ff
                do
                    for lam in $lambda
                    do
                        echo $mod $f $lam
                        python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -ff $f -lambda $lam -noplot
                    done
                done
            done
        done


    elif [ $analysis == 'reg' ]; then
        echo 'Regularization analysis'

        reg='L1 L2 smooth smooth_simple'
        mu='10 5 3 1 0.75 0.5 0.25 0.1 0.05 0.01 0.001 0.0001 0.00001 0'
        oricamod='A_block W_block'
        bl=50
        ff='constant'
        lambda='N'


        for r in $recordings
        do
            for mod in $oricamod
            do
                for re in $reg
                do
                    for m in $mu
                    do
                        echo $mod $re $m
                        python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -ff $ff -lambda $lambda -reg $re -mu $m -noplot
                    done
                done
            done
        done

    elif [ $analysis == 'ica' ]; then
        echo 'Running FastICA'

        for r in $recordings
        do
            echo $r ICA
        done
    fi
fi
#!/usr/bin/env bash

cwd=$(pwd)
ff=cooling
oricamod='A_block W_block original'
chdir=$PWD

if [ $# == 0 ]; then
    echo "Supply rec electrode folder (includeing final /), mode (block - ff - reg)"
elif [ $# == 2 ]; then
    folder=$1
    analysis=$2
    recordings=$(ls $folder)

    if [ $analysis == 'block' ]; then
        #blocks='1 5 10 20 35 50 75 100 200 350 500 750 1000 1500 2000'
        blocks='1 5 10 20  50  100  500 1000  2000'

        echo 'Block analysis'


        for r in $recordings
        do
            for mod in $oricamod
            do
                for bl in $blocks
                do
                    echo $mod $bl
                    python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -noplot -resfile results_block.csv
                done
            done
        done

    elif [ $analysis == 'ff' ]; then
        echo 'FF analysis'
        ff='cooling constant'
        #lambda='N 5 3 0.995 0.75 0.5 0.25 0.1 0.01 0.001 0.0001 0.00001'
        lambda='N 5  3 0.995 0.5  0.1 0.005 0.00001'
        bl=50

        for r in $recordings
        do
            for mod in $oricamod
            do
                for f in $ff
                do
                    for lam in $lambda
                    do
                        echo $mod $f $lam
                        python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -ff $f -lambda $lam -noplot -resfile results_ff.csv
                    done
                done
            done
        done


    elif [ $analysis == 'reg' ]; then
        echo 'Regularization analysis'

        reg='L1 L2 smooth smooth_simple'
        mu='10 5 3 1 0.75 0.5 0.25 0.1 0.05 0.01 0.001 0.0001 0.00001 0'
        oricamod='A_block W_block'
        bl=50
        ff='constant'
        lambda='N'


        for r in $recordings
        do
            for mod in $oricamod
            do
                for re in $reg
                do
                    for m in $mu
                    do
                        echo $mod $re $m
                        python ../evaluate_ICA.py -r $folder$r -oricamod $mod -block $bl -ff $ff -lambda $lambda -reg $re -mu $m -noplot -resfile results_reg.csv
                    done
                done
            done
        done

    elif [ $analysis == 'ica' ]; then
        echo 'Running FastICA'

        for r in $recordings
        do
            python ../evaluate_ICA.py -r $folder$r -mod ica -noplot -resfile results_ica.csv
        done
    elif [ $analysis == 'remove' ]; then
        echo 'Running FastICA'

        for r in $recordings
        do
            cd $folder$r
            rm results.csv
            echo 'removing '  $r  'results.csv'
            cd ..
        done
    fi
fi
