#!/bin/bash
sudo nvidia-docker run --rm -ti --volume=$(pwd):/SelFlow:rw --workdir=/SelFlow --ipc=host tensorflow/tensorflow:1.8.0-gpu-py3 /bin/bash

