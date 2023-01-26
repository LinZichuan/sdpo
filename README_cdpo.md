## Sample Dropout in Policy Optimization

## How to use
#Creating image first,build docker image
cd docker_for_tpami
sh build.sh 

#run experiments in batch
# set env and seed nums, and select the corresponding algorithm to use, E.g:
vim tasks_espo_cd.txt 

# or in another way
vim change_env.sh

# start batch experiements
nohup sh train_espo_cd.sh &

#zip or unzip log files
sh zipit.sh
sh unzipit.sh

# plot result
sh paint.sh

# kill 
pkill python

