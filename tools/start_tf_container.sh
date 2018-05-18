#!/bin/bash

nvidia-docker kill tf_gpu
nvidia-docker rm tf_gpu

set -e

nvidia-docker build -t tf-1.7 docker
nvidia-docker run -d \
	-p 8888:8888 \
	-p 0.0.0.0:6006:6006 \
	--name tf_gpu \
	-v "$(pwd)/src":"/app" \
	-v "$(pwd)/models":"/models" \
	-v "$(pwd)/datasets":"/datasets" \
	-v "/usr/local/cuda/extras/CUPTI/lib64":"/usr/local/cuda/extras/CUPTI/lib64" \
	tf-1.7