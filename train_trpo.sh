#game=$1
#echo $game
export LOGDIR='log_trpo'
cat tasks_trpo.txt | xargs -n 4 -P 10 \
    sh -c 'OMPI_MCA_btl_vader_single_copy_mechanism=none OMP_NUM_THREADS=10 python -m baselines.run --env=$0 --config=$1 --seed=$2 --num_timesteps=$3'
#game=$1
#echo $game
#nohup python -m baselines.run --env=$game --config=trpo.yaml --seed=0 --num_timesteps=10000000 > ./alllog/$game_0.log &
#sleep 2
#nohup python -m baselines.run --env=$game --config=trpo.yaml --seed=1 --num_timesteps=10000000 > ./alllog/$game_1.log &
#sleep 2
