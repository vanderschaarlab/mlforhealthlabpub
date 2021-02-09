# Dockerfile example 1b.

# This Dockerfile will set up the same dependencies as the competition backend.
# Includes tensorflow 1.15.2 and keras 2.3.1.
# CPU-only version.

FROM tensorflow/tensorflow:1.15.2

RUN apt-get update -qy
RUN apt-get install -qy python3-pip python3-numpy
RUN pip3 install --upgrade pip

RUN pip3 install pandas numpy tqdm scikit-learn keras==2.3.1
