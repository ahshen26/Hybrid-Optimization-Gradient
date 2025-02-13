#!/usr/bin/bash
########################################################
# job_builder.sh
# Job Arrays without the Resource Manager 
# Version 1.0.0
########################################################

# log hazard ratio between TRT and CONTROL
for i in 0.3516406 0;
    do
    # log hazard ratio between CONTROL and RWD
    for j in `seq -0.7 0.1 0.7`;
        do
        # iteration
        for n in `seq 1 1 20`;
            do
            # echo $i $j $n
            sbatch Simulation_Hybrid.sub $i $j $n
            done
        done
    done