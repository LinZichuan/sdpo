mkdir alllog > /dev/null
for game in PongNoFrameskip-v4 AlienNoFrameskip-v4 AmidarNoFrameskip-v4
do
    sh train_trpo.sh $game
    sh train_trpo_cd.sh $game
done