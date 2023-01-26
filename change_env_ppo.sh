#env="Humanoid-v2"
env="dm_dog.fetch-v0"
sed -i "s/.* ppo/$env ppo/g" tasks_ppo.txt
sed -i "s/.* ppo/$env ppo/g" tasks_ppo_cd.txt
#sd_delta=0.25
#sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd.yaml
#sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_noabs.yaml
#sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_value.yaml
#sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_unbias.yaml
#sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_lclip.yaml
