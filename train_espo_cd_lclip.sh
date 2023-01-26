export LOGDIR='log_espo_cd_lclip'
cat tasks_espo_cd_lclip.txt | xargs -n 4 -P 10 \
    sh -c 'python -m baselines.run --env=$0 --config=$1 --seed=$2 --num_timesteps=$3'
