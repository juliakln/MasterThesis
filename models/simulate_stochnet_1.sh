#!/bin/bash

i=111
for bees in {15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150}
do
	python StochNetV2/stochnet_v2/scripts/simulate_histogram_data_gillespy.py --project_folder $1 --nb_trajectories 100 --endtime 50 --dataset_id ${i} --nbees ${bees} --ph ${bees}
	i=$(($i+1))
done