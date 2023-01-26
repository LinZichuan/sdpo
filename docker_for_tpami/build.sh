#!/bin/bash
#可以参考开悟任务号:191209 
dockerfile="Dockerfile"
image_name="mirrors.tencent.com/aiarena/rl_baselines_atari"
docker build -t ${image_name} -f $dockerfile --network=host . 
docker push ${image_name}
