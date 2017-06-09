#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lm/kaldi-rnnlm.h"
#include "lm/mikolov-rnnlm-lib.h"
#include "util/common-utils.h"

using namespace tensorflow;

int main(int argc, char* argv[]) {
//*
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  const string pathToGraph = "/export/b02/hxu/TensorFlow/kaldi/egs/ami/s5/data/tensorflow/model.small.meta";
  const string checkpointPath = "/export/b02/hxu/TensorFlow/kaldi/egs/ami/s5/data/tensorflow/model.small";

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  MetaGraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkpointPath;
  
  status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Setup inputs and outputs:
  std::vector<Tensor> state;
//  std::vector<Tensor> state(DT_FLOAT, {2, 2, 1, 200});
  status = session->Run(std::vector<std::pair<string, tensorflow::Tensor>>(), {"Train/Model/test_initial_state"}, {}, &state);

  for (int32 word_out = 0; word_out < 10000; word_out++) {
    Tensor in_word(DT_INT32, {1, 1});
    in_word.scalar<int32>()() = (word_out + 9999) % 10000; 

    Tensor out_word(DT_INT32, {1, 1});
    out_word.scalar<int32>()() = word_out; 

    // num-layers
    // 2 (c and h)
    // 1 (batchsize)
    // hidden-size

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"Train/Model/test_word_in", in_word},
      {"Train/Model/test_word_out", out_word},
      {"Train/Model/test_state", state[0]},
    };

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> outputs;

    // Run the session, evaluating our "c" operation from the graph
    status = session->Run(inputs, {"Train/Model/test_out", "Train/Model/test_state_out"}, {}, &outputs);

    if (!status.ok()) {
      std::cout << status.ToString() << "\n";
      return 1;
    }

    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.

    // (There are similar methods for vectors and matrices here:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

    // Print the results
    std::cout << word_out << ": " << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
    std::cout << word_out << ": " << outputs[1].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
    state[0] = outputs[1];
//    std::cout << output_c() << "\n"; // 30
  }

  // Free any resources used by the session
  session->Close();
  // */
  return 0;
}
