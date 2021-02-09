#!/bin/bash

# VERSION:
TAG="latest"

# Set these:
USER="drshushen"
REPO="hide-and-seek-codalab"

docker login --username=$USER
docker build -t $USER/$REPO:$TAG . 
docker push $USER/$REPO:$TAG
