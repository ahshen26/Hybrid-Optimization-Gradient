#!/usr/bin/bash
########################################################
# job_builder.sh
# Job Arrays without the Resource Manager 
# Version 1.0.0
########################################################

#Randomization Ratio = 0.5, 0.865
for r in 1 3;
	do
	# log hazard ratio between TRT and CONTROL
	for i in -0.4 0;
    		do
    		# log hazard ratio between CONTROL and RWD
    		for j in `seq -0.7 0.1 0.7`;
        		do
        		# iteration
        		for n in `seq 1 1 20`;
            			do
            			# echo $r $i $j $n
            			sbatch Simulation_Hybrid.sub $r $i $j $n
            			done
        		done
    		done
	done
