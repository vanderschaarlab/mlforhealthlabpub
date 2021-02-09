Alternative workers
===================

### [igormusinov/competitions-v1-compute-worker](https://github.com/igormusinov/competitions-v1-compute-worker)

Uses cool Azure features ([ACI](https://azure.microsoft.com/en-us/services/container-instances/)) to run compute worker docker container in serverless environment:

### [(LEGACY nvidia-docker) nvidia compute worker](https://github.com/codalab/competitions-v1-compute-worker/tree/162-nvidia-worker)

Adds support for nvidia GPUs 

### [realtime detailed results](https://github.com/codalab/competitions-v1-compute-worker/tree/feature/realtime-detailed-results)

Adds support for real time detailed results

Running
=======

### If you want to run everything in one line:

*Note: this will make a `/tmp/codalab` directory*

```
mkdir -p /tmp/codalab && docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/codalab:/tmp/codalab \
    -d \
    --name compute_worker \
    --env BROKER_URL=<queue broker url> \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v1-compute-worker:latest
```


### If you want to run using `.env` configuration:

Edit `.env_sample` and save it as `.env`

Make sure the temp directory you select is created and pass it in this command

```
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/codalab:/tmp/codalab \
    -d \
    --name compute_worker \
    --env-file .env \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v1-compute-worker:latest
```

### If you want to run with GPU:

Make sure that you have [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
set up -- this also involves updating to Docker 19.03 and installing NVIDIA drivers.

Edit `.env_sample` and save it as `.env`. Make sure to uncomment `USE_GPU=True`.

Then make sure the temp directory you select is created and pass it in this command

```
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/codalab:/tmp/codalab \
    -d \
    --name compute_worker \
    --env-file .env \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    --gpus all \
    codalab/competitions-v1-compute-worker:latest
```

### To get output of the worker

```
$ docker logs -f compute_worker
```

### To stop the worker

```
$ docker kill compute_worker
```


Development
===========

To re-build the image:

```
docker build -t competitions-v1-compute-worker .
```

Updating the image

```
docker build -t codalab/competitions-v1-compute-worker:latest .
docker push codalab/competitions-v1-compute-worker
```


Special env flags
=================

### USE_GPU

*Default False, does not pass `--gpus all` flag*

Note: Also requires Docker v19.03 or greater, nvidia-container-toolkit, and NVIDIA drivers.

### SUBMISSION_TEMP_DIR

*Default /tmp/codalab*

### SUBMISSION_CACHE_DIR

*Default /tmp/cache*

### CODALAB_HOSTNAME

*Default socket.gethostname()*

### DONT_FINALIZE_SUBMISSION

*Default False*

Sometimes it may be useful to pause the compute worker and return instead of finishing a submission. This leaves the
submission in a state where it hasn't been cleaned up yet and you can attempt to re-run it manually.
