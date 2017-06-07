#!/bin/bash

set -e

export HOME=/home/hainanx/work
#export JAVA_HOME=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121
#export PATH=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121/bin/:$PATH
export PATH=$PWD/bazel/output/:$PATH
#export PATH=$PWD/tensorflow/bazel-out/host/bin/external/protobuf/:$PATH
export PATH=$PWD:$PATH

#git clone https://github.com/tensorflow/tensorflow
#[ ! -f bazel.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-dist.zip -O bazel.zip
#mkdir -p bazel
#cd bazel
#unzip ../bazel.zip
#./compile.sh
#cd ../

# now bazel is built
#git clone https://github.com/tensorflow/tensorflow

cd tensorflow

#./configure

#bazel build //tensorflow/core:framework_headers_lib
#
#bazel build //tensorflow:libtensorflow.so
bazel build //tensorflow:libtensorflow_cc.so

#exit
#
#cd tensorflow/tensorflow
#mkdir -p rnnlm
#cd rnnlm
#
#[ ! -f BUILD ] && ln -s ../../../../src/tensorflow/BUILD
#[ ! -f WORKSPACE ] && ln -s ../../../../src/tensorflow/WORKSPACE
#[ ! -f loader_rnn.cc ] && ln -s ../../../../src/tensorflow/loader_rnn.cc
#[ ! -d kaldi_src ] && ln -s ../../../../src/ kaldi_src
#
#bazel build --test_tmpdir=$TEST_TMPDIR :loader_rnn
##bazel run -c opt :loader_rnn
#
