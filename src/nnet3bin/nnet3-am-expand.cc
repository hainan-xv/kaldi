// nnet3bin/nnet3-am-copy.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2016 Daniel Galvez

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <typeinfo>
#include <sstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy nnet3 neural-net acoustic model file; supports conversion\n"
        "to raw model (--raw=true).\n"
        "Also supports setting all learning rates to a supplied\n"
        "value (the --learning-rate option),\n"
        "and supports replacing the raw nnet in the model (the Nnet)\n"
        "with a provided raw nnet (the --set-raw-nnet option)\n"
        "\n"
        "Usage:  nnet3-am-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-copy --binary=false 1.mdl text.mdl\n"
        " nnet-am-copy --raw=true 1.mdl 1.raw\n";

    bool binary_write = true,
        raw = false;
    BaseFloat learning_rate = -1;
    std::string set_raw_nnet = "";
    std::leaves_string = "";
    bool convert_repeated_to_block = false;
    BaseFloat scale = 1.0;
    int32 num_outputs;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("raw", &raw, "If true, write only 'raw' neural net "
                "without transition model and priors.");
    po.Register("set-raw-nnet", &set_raw_nnet,
                "Set the raw nnet inside the model to the one provided in "
                "the option string (interpreted as an rxfilename).  Done "
                "before the learning-rate is changed.");
    po.Register("convert-repeated-to-block", &convert_repeated_to_block,
                "Convert all RepeatedAffineComponents and "
                "NaturalGradientRepeatedAffineComponents to "
                "BlockAffineComponents in the model. Done after set-raw-nnet.");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of updatable components"
                " are set to this value.");
    po.Register("scale", &scale, "The parameter matrices are scaled"
                " by the specified value.");
    po.Register("leaves", &leaves_string, "leaves string");
    po.Register("num-outputs", &num_outputs, "number of outputs");


    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);

    std::vector<std::string> trans_model_rxfilenames;
    int i = 3;
    for (; i <= po.NumArgs();) {
      trans_model_rxfilenames.push_back(po.GetArg(i++));
    }
    KALDI_ASSERT(trans_model_rxfilenames.size() == num_outputs);

    vector<TransitionModel*> trans_models;
    trans_models.resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      trans_models[i] = new TransitionModel();
      ReadKaldiObject(trans_model_rxfilenames[i], trans_models[i]);
    }

    TransitionModel old_trans_model;
    AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      old_trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    stringstream ss(leaves_string);
    vector<int32> leaves_num;
    int32 n;
    while (ss >> n) {
      leaves_num.push_back(n);
    }
    KALDI_ASSERT(leaves_num.size() == num_outputs);

    AmNnetMulti am_nnet_out(am_nnet, leaves_num);

    if (!set_raw_nnet.empty()) {
      Nnet nnet;
      ReadKaldiObject(set_raw_nnet, &nnet);
      am_nnet.SetNnet(nnet);
    }

    if (raw) {
      WriteKaldiObject(am_nnet.GetNnet(), nnet_wxfilename, binary_write);
      KALDI_LOG << "Copied neural net from " << nnet_rxfilename
                << " to raw format as " << nnet_wxfilename;

    } else {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Copied neural net from " << nnet_rxfilename
                << " to " << nnet_wxfilename;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
