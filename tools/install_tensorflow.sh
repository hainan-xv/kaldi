#!/bin/bash

set -e

export HOME=/export/b02/hxu
export JAVA_HOME=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121
export PATH=/export/b02/hxu/TensorFlow/java/jdk1.8.0_121/bin/:$PATH

git clone https://github.com/tensorflow/tensorflow


[ ! -f bazel-0.4.5-dist.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-dist.zip
mkdir -p bazel
cd bazel
unzip ../bazel-0.4.5-dist.zip
./compile.sh
cd ../

# now bazel is built

export PATH=$PWD/bazel/output/:$PATH

cd tensorflow

./configure

cd ../

cd tensorflow/tensorflow
mkdir -p rnnlm
cd rnnlm

[ ! -f BUILD ] && ln -s ../../../../src/tensorflow/BUILD
[ ! -f loader_rnn.cc ] && ln -s ../../../../src/tensorflow/loader_rnn.cc

TEST_TMPDIR=tensorflow/build

echo bazel build :loader_rnn
bazel build --test_tmpdir=$TEST_TMPDIR :loader_rnn
bazel run -c opt :loader_rnn













