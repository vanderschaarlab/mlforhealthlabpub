#!/bin/bash

# Description:
# Helper script for building and uploading a docker image.
# Requires that docker is installed on the system.

# Set these:
USER="ENTER YOUR DOCKERHUB USERNAME"
REPO="ENTER DOCKERHUB REPO NAME TO CREATE"
DOCKERFILE="ENTER PATH TO YOUR Dockerfile"  # Path to Dockerfile to build from, e.g. "./example1a"
TAG="latest"  # The tag (version) of the docker image to create, default is "latest".



# ----------------------------------------------------------------------------------------------------------------------

docker login --username=$USER
docker build -t $USER/$REPO:$TAG $DOCKERFILE
docker push $USER/$REPO:$TAG
