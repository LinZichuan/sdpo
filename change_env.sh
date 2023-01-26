env="Humanoid-v2"
sd_delta=0.25
sed -i "s/.* ppo/$env ppo/g" tasks_espo.txt
sed -i "s/.* ppo/$env ppo/g" tasks_espo_cd.txt
sed -i "s/.* ppo/$env ppo/g" tasks_espo_cd_noabs.txt
sed -i "s/.* ppo/$env ppo/g" tasks_espo_cd_value.txt
sed -i "s/.* ppo/$env ppo/g" tasks_espo_cd_unbias.txt
sed -i "s/.* ppo/$env ppo/g" tasks_espo_cd_lclip.txt
sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd.yaml
sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_noabs.yaml
sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_value.yaml
sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_unbias.yaml
sed -i "s/sd_delta: .*/sd_delta: $sd_delta/g" configs/ppo_dropout_cd_lclip.yaml
