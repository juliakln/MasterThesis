#!/bin/bash

i=111
for bees in {15,20,25,30,35,40,45,50,55,60,65,70,75,80,85}
do
	for rate in {0.01,0.01428571,0.01857143,0.02285714,0.02714286,0.03142857,0.03571429,0.04,0.04428571,0.04857143,0.05285714,0.05714286,0.06142857,0.06571429,0.07}
	do
		python StochNetV2/stochnet_v2/scripts/simulate_histogram_data_gillespy.py --project_folder $1 --nb_trajectories 1000 --endtime 50 --dataset_id ${i} --nbees ${bees} --ph ${bees} --k1 ${rate}
		i=$(($i+1))
	done
done