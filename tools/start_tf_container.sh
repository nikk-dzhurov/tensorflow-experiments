#!/bin/bash

docker kill tf_gpu
docker rm tf_gpu

set -e

docker build -t local-tf docker
docker run -d \
	--gpus all \
	--user=1000:1000 \
	-p 8888:8888 \
	-p 3000:3000 \
	-p 0.0.0.0:6006:6006 \
	--name tf_gpu \
	-v "$(pwd)/src":"/app" \
	-v "$(pwd)/data":"/data" \
	-v "$(pwd)/models":"/models" \
	-v "$(pwd)/datasets":"/datasets" \
	-v "$(pwd)/test_images":"/test_images" \
	-v "$(pwd)/samples":"/samples" \
	-v "/usr/local/cuda/extras/CUPTI/lib64":"/usr/local/cuda/extras/CUPTI/lib64" \
	local-tf
