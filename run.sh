#!/bin/bash

docker exec -it \
	--workdir /app \
	tf_gpu "$@"
