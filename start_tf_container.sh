#!/bin/bash

docker kill tf_gpu
nvidia-docker rm tf_gpu
nvidia-docker run -d \
	-p 8888:8888 \
	-p 0.0.0.0:6006:6006 \
	--env-file=vars.env \
	--name tf_gpu \
	-v "$(pwd)/src":"/app" \
	-v "$(pwd)/models":"/models" \
	-v "$(pwd)/datasets":"/datasets" \
	-v "/usr/local/cuda/extras/CUPTI/lib64":"/usr/local/cuda/extras/CUPTI/lib64" \
	tensorflow/tensorflow:1.7.0-gpu-py3
