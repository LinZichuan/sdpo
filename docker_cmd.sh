#!/usr/bin/env bash
docker run --rm --user $(id -u) \
                -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
                -v `pwd`:/home/mingfei/ppo \
                --entrypoint '/bin/bash' ppo_tf:1.0 \
                -c "OMP_NUM_THREADS=10 $1"
