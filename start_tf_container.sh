#!/bin/bash

docker kill tf_gpu
nvidia-docker rm tf_gpu
nvidia-docker run -d \
	-p 8888:8888 \
	-v "$(pwd)/src":"/app" \
	-v "$(pwd)/models":"/models" \
	--name tf_gpu \
	tensorflow/tensorflow:latest-gpu
