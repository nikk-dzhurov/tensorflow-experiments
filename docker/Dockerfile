FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install flask Pillow

CMD ["/bin/bash",  "-c", "while true; do sleep 100; done"]
