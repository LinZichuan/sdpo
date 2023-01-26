#!/usr/bin/env bash
nohup docker build --build-arg UID=$UID -t ppo_tf:1.0 . &