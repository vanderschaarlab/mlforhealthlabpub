#
FROM tensorflow/tensorflow:1.15.2-gpu

RUN apt-get update -qy
RUN apt-get install -qy python3-pip python3-numpy
RUN pip3 install --upgrade pip

RUN pip3 install docker pandas numpy tqdm scikit-learn keras==2.3.1 tensorflow-determinism
