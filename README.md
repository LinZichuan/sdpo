## Ratio clipping in PPO

## How to use
Creating image first
```bash 
## build docker image
cd docker
bash docker_build.sh # check the building process by `cat nohup.out` 
```

Run experiements in debug mode
```bash 
## start a container first
bash docker_shell.sh

# run the code
python3 -m baselines.run --env=Hopper-v2 --config=default.yaml # using default config in configs folder
```

Run experiments in batch
```bash 
## check all the tasks in tasks.txt and specify the gpu to use

## check train.sh and specify the number of tasks to run in parallel
# 
#
# cat tasks.txt | xargs -n 5 -p 70 sh -c ...
#                                ^---> this number specifies the number of tasks to run in parrallel 
#                          ^--> this number should be consistent with the setup (i.e., number of columns) in tasks.txt

# finally run all exps specificed in tasks
bash train.sh
```

Clean up all exp contains or stop all experiments
```bash
# Ctrl-C to the terminal where the batch experiments started

bash kill.sh # this will kill all exp containers
```
