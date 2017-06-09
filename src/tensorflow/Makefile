include ../kaldi.mk

TENSORFLOW = ../../tools/tensorflow
KALDI_ROOT = ../../
all:

#EXTRA_CXXFLAGS = -Wno-sign-compare -fPIC -I ../ -I $(KALDI_ROOT)/src -DKALDI_NO_PORTAUDIO -I $(TENSORFLOW)/third_party/eigen3 -I $(BAZEL)/third_party/protobuf/3.0.0/src/ \
#                 -I$(TENSORFLOW)/bazel-genfiles -I $(TENSORFLOW) \
#  -I $(TENSORFLOW)/tensorflow/contrib/makefile/gen/protobuf/include/ -I $(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen \

EXTRA_CXXFLAGS = -Wno-sign-compare -fPIC -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen/
#EXTRA_CXXFLAGS = -Wno-sign-compare -fPIC -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) -I$(TENSORFLOW)/third_party/eigen3

BINFILES = loader_rnn

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../../tools/tensorflow/bazel-bin/tensorflow/tensorflow_cc.so

LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += $(OTHERLIBS) -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc

include ../makefiles/default_rules.mk
