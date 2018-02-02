#!/bin/bash

export HOME=$PWD/tensorflow_build/

gpu=true

tf_source=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl

if ! $gpu; then
  tf_source=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl
fi

pip install --user $tf_source
