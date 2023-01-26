#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} run --rm \
           --user $(id -u) \
           -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
           -v `pwd`:/home/mingfei/ppo \
           -it ppo_tf:1.0
