#!/bin/bash

export JAVA_HOME=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121
export PATH=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121/bin/:$PATH

#git clone https://github.com/tensorflow/tensorflow

#cd tensorflow
#
#git checkout r1.0
#
#cd ../
#
##git clone https://github.com/google/bazel/

[ ! -f bazel-0.4.5-dist.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-dist.zip
mkdir -p bazel
cd bazel

unzip ../bazel-0.4.5-dist.zip

./compile.sh

#mkdir build
#./compile.sh compile build/















