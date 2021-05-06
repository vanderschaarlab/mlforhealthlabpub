#!/bin/bash

set -eu

docker build . -t competitions-v1-compute-worker

mkdir -p /tmp/codalab

sudo mkdir -p /opt/hide-and-seek/
sudo mkdir -p /opt/hide-and-seek/data
sudo mkdir -p /opt/hide-and-seek/hiders
sudo mkdir -p /opt/hide-and-seek/seekers
sudo mkdir -p /opt/hide-and-seek/input
sudo mkdir -p /opt/hide-and-seek/scoring

docker run \
       -v /opt/hide-and-seek:/opt/hide-and-seek \
       -v /var/run/docker.sock:/var/run/docker.sock \
       -v /tmp/codalab:/tmp/codalab \
       -d \
       --name compute_worker \
       --env-file .env \
       --restart unless-stopped \
       --log-opt max-size=50m \
       --log-opt max-file=3 \
       competitions-v1-compute-worker  # Using the local, customized image.
