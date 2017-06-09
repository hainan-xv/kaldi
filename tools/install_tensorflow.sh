#!/bin/bash

set -e

export HOME=/export/b02/hxu
export JAVA_HOME=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121
export PATH=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121/bin/:$PATH
export PATH=$PWD/bazel/output/:$PATH

[ ! -f bazel.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-dist.zip -O bazel.zip
mkdir -p bazel
cd bazel
unzip ../bazel.zip
./compile.sh
cd ../

## now bazel is built
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure

tensorflow/contrib/makefile/download_dependencies.sh 
bazel build //tensorflow:libtensorflow.so
#bazel build //tensorflow:libtensorflow_cc.so
