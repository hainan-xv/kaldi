// latbin/lattice-lmrescore-tf-rnnlm.cc

// Copyright 2017  Hainan Xu

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


#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "tensorflow/tensorflow-rnnlm-lib.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::tf_rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Rescores lattice with rnnlm. The LM will be wrapped into the\n"
        "DeterministicOnDemandFst interface and the rescoring is done by\n"
        "composing with the wrapped LM using a special type of composition\n"
        "algorithm. Determinization will be applied on the composed lattice.\n"
        "\n"
        "Usage: lattice-lmrescore-tf-rnnlm [options] [unk-file] <rnnlm-wordlist> \\\n"
        "             <word-symbol-table-rxfilename> <lattice-rspecifier> \\\n"
        "             <rnnlm-rxfilename> <lattice-wspecifier>\n"
        " e.g.: lattice-lmrescore-tf-rnnlm --lm-scale=-1.0 unkcounts.txt rnnwords.txt \\\n"
        "              words.txt ark:in.lats rnnlm ark:out.lats\n";

    ParseOptions po(usage);
    int32 max_ngram_order = 3;
    BaseFloat lm_scale = 1.0;
    int32 length = -1;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs; frequently 1.0 or -1.0");
    po.Register("max-ngram-order", &max_ngram_order, "If positive, limit the "
                "rnnlm context to the given number, -1 means we are not going "
                "to limit it.");
    po.Register("utt-id-compare-length", &length, "How many characters we look "
                "at when deciding if 2 utt are from the same recording");


    KaldiTfRnnlmWrapperOpts opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    if (length == -1) {
      KALDI_ERR << "You have to set --length to be a positive number";
    }

    std::string lats_rspecifier, rnn_word_list,
      word_symbols_rxfilename, rnnlm_rxfilename, lats_wspecifier, unk_prob_file;
    if (po.NumArgs() == 5) {
      rnn_word_list = po.GetArg(1);
      word_symbols_rxfilename = po.GetArg(2);
      lats_rspecifier = po.GetArg(3);
      rnnlm_rxfilename = po.GetArg(4);
      lats_wspecifier = po.GetArg(5);
    } else {
      unk_prob_file = po.GetArg(1);
      rnn_word_list = po.GetArg(2);
      word_symbols_rxfilename = po.GetArg(3);
      lats_rspecifier = po.GetArg(4);
      rnnlm_rxfilename = po.GetArg(5);
      lats_wspecifier = po.GetArg(6);
    }

    // Reads the TF language model.
    KaldiTfRnnlmWrapper rnnlm(opts, rnn_word_list, word_symbols_rxfilename,
                                unk_prob_file, rnnlm_rxfilename);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;
    std::string last_key = "";
    bool is_same_recording = false;

    tensorflow::Tensor initial_context, initial_cell;

    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
//      KALDI_LOG << "key is " << key;
      if (last_key.size() >= length || key.substr(0, length) == last_key.substr(0, length)) {
        // same recording
        is_same_recording = true;
//        KALDI_LOG << "same";
      } else {
        is_same_recording = false;
//        KALDI_LOG << "diff";
      }

      last_key = key;

      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();

      if (lm_scale != 0.0) {
        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Wraps the rnnlm into FST. We re-create it for each lattice to prevent
        // memory usage increasing with time.

        // TODO if key corresponds to new recording, initialize with default
        //      otherwise initialize with last query result in 3
        if (is_same_recording) {
          rnnlm.SetInitialContext(initial_context);
          rnnlm.SetInitialCell(initial_cell);
        }

        TfRnnlmDeterministicFst rnnlm_fst(max_ngram_order, &rnnlm);

        // Composes lattice with language model.
        CompactLattice composed_clat;
        ComposeCompactLatticeDeterministic(clat, &rnnlm_fst, &composed_clat);

        // Determinizes the composed lattice.
        Lattice composed_lat;
        ConvertLattice(composed_clat, &composed_lat);
        Invert(&composed_lat);
        CompactLattice determinized_clat;
        DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), &determinized_clat);

        if (determinized_clat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key
              << " (incompatible LM?)";
          n_fail++;
        } else {
          compact_lattice_writer.Write(key, determinized_clat);
          n_done++;
        }

        // TODO 1. compute the best-path under one setting
        fst::ScaleLattice(fst::LatticeScale(1, 0.1), &determinized_clat);
        CompactLattice clat_best_path;
        CompactLatticeShortestPath(determinized_clat, &clat_best_path);
        Lattice best_path;
        ConvertLattice(clat_best_path, &best_path);

        std::vector<int32> inputs, outputs;
        LatticeWeight weight;
        GetLinearSymbolSequence(best_path, &inputs, &outputs, &weight);

        // TODO 2. get the last n words in the best-path (n is the ngram order)
        std::vector<int32> ngram_of_last_utt;

        std::stringstream best_seq;
        std::stringstream ngram_seq;
        for (int i = 0; i < outputs.size(); i++) {
          best_seq << outputs[i] << ", ";
          if (outputs.size() - i < max_ngram_order) {
            ngram_of_last_utt.push_back(outputs[i]);
            ngram_seq << outputs[i] << ", ";
          }
        }
//        KALDI_LOG << key <<": output is" << best_seq.str();
//        KALDI_LOG << key <<": ngram is" << ngram_seq.str();

        // TODO 3. Query into the rnnlm_fst and find the associate hidden states
        rnnlm_fst.GetContextFromNgram(ngram_of_last_utt, &initial_context, &initial_cell);

      } else {
        // Zero scale so nothing to do.
        n_done++;
        compact_lattice_writer.Write(key, clat);
      }
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
