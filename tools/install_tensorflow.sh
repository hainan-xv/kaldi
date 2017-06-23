#!/bin/bash

set -e

export HOME=$PWD/tensorflow_build/
mkdir -p $HOME

[ ! -f bazel.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-dist.zip -O bazel.zip
mkdir -p bazel
cd bazel
unzip ../bazel.zip
./compile.sh
cd ../

# now bazel is built
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure

tensorflow/contrib/makefile/download_dependencies.sh 
bazel build --copt=-msse4.2 //tensorflow:libtensorflow.so
bazel build --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
