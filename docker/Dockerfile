FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install h5py==2.8.0rc1
RUN python3 -m pip install flask

CMD ["/bin/bash",  "-c", "while true; do sleep 100; done"]