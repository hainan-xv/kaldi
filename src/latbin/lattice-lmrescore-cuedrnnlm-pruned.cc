// latbin/lattice-lmrescore-cuedrnnlm.cc

// Copyright 2016  Ricky Chan Ho Yin

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
#include "lm/kaldi-rnnlm.h"
#include "util/common-utils.h"
#include "cuedlmcpu/helper.h"
#include "lat/compose-lattice-pruned.h"
#include <fstream>

namespace kaldi {

fst::VectorFst<fst::StdArc> *ReadAndPrepareLmFst(std::string rxfilename) {
  // ReadFstKaldi() will die with exception on failure.
  fst::VectorFst<fst::StdArc> *ans = fst::ReadFstKaldi(rxfilename);
  if (ans->Properties(fst::kAcceptor, true) == 0) {
    // If it's not already an acceptor, project on the output, i.e. copy olabels
    // to ilabels.  Generally the G.fst's on disk will have the disambiguation
    // symbol #0 on the input symbols of the backoff arc, and projection will
    // replace them with epsilons which is what is on the output symbols of
    // those arcs.
    fst::Project(ans, fst::PROJECT_OUTPUT);
  }
  if (ans->Properties(fst::kILabelSorted, true) == 0) {
    // Make sure LM is sorted on ilabel.
    fst::ILabelCompare<fst::StdArc> ilabel_comp;
    fst::ArcSort(ans, ilabel_comp);
  }
  return ans;
}


}  // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using fst::ReadFstKaldi;
    using std::unique_ptr;

    const char *usage =
        "Rescores lattice with cued rnnlm. The LM will be wrapped into the\n"
        "DeterministicOnDemandFst interface and the rescoring is done by\n"
        "composing with the wrapped LM using a special type of composition\n"
        "algorithm. Determinization will be applied on the composed lattice.\n"
        "\n"
        "Usage: lattice-lmrescore-cuedrnnlm [options] \\\n"
        "             <word-symbol-table-rxfilename> <lattice-rspecifier> \\\n"
        "             <rnnlm-rxfilename> <lattice-wspecifier> \\\n"
        "             <input-wordlist> <output-wordlist> <rnnlm-info>\n"
        " e.g.: lattice-lmrescore-cuedrnnlm --lm-scale=-1.0 words.txt \\\n"
        "                     ark:in.lats rnnlm ark:out.lats \\\n"
        "                     inputwordlist outputwordlist rnnlm-info\n";

    ParseOptions po(usage);
    int32 max_ngram_order = 3;
    BaseFloat lm_scale = 1.0;
    int32 nthread = 1;
    BaseFloat acoustic_scale = 0.1;

    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "costs; frequently 1.0 or -1.0");
    po.Register("max-ngram-order", &max_ngram_order, "If positive, limit the "
                "rnnlm context to the given number, -1 means we are not going "
                "to limit it.");
    po.Register("nthread", &nthread, "number of thread for cued rnnlm run with OpenMP, default is 1 ");

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic "
                "probabilities (e.g. 0.1 for non-chain systems); important because "
                "of its effect on pruning.");

    KaldiRnnlmWrapperOpts opts;
    ComposeLatticePrunedOptions compose_opts;
    opts.Register(&po);
    compose_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string lm_to_subtract_rxfilename, lats_rspecifier, unk_prob_rspecifier,
        word_symbols_rxfilename, rnnlm_rxfilename, lats_wspecifier;

    std::string inputwordlist, outputwordlist, rnnlminfofile;

    unk_prob_rspecifier = "";
    lm_to_subtract_rxfilename = po.GetArg(1),
    word_symbols_rxfilename = po.GetArg(2);
    lats_rspecifier = po.GetArg(3);
    rnnlm_rxfilename = po.GetArg(4);  // cuedrnnlm rnnlm.txt model v0.1
    lats_wspecifier = po.GetArg(5);
    inputwordlist = po.GetArg(6);   // cuedrnnlm inputwordlist
    outputwordlist = po.GetArg(7);  // cuedrnnlm outputwordlist
    rnnlminfofile = po.GetArg(8);   // e.g. -layers 172887:200:172887 -fullvocsize 175887

    KALDI_LOG << "Reading old LMs...";
    VectorFst<StdArc> *lm_to_subtract_fst = ReadAndPrepareLmFst(
        lm_to_subtract_rxfilename);
    fst::BackoffDeterministicOnDemandFst<StdArc>
              lm_to_subtract_det_backoff(*lm_to_subtract_fst);
    fst::ScaleDeterministicOnDemandFst
              lm_to_subtract_det_scale(-lm_scale, &lm_to_subtract_det_backoff);

    string str;
    vector<int> layersizes;
    int fullvocsize;

    ifstream infile(rnnlminfofile.c_str(), ios::in);
    infile >> str; // -layers
    infile >> str;
#if 0
    if(str.compare("-layers")) {
      cout << "rnnlm-info format incorrect for -layers";
      exit(1);
    }
    parseArray(str, layersizes);
    for(int i=0; i<layersizes.size(); i++) {
      if(layersizes[i]<=0) {
        cout << "rnnlm-info value incorrect for -layers";
        exit(1);
      }
    }
#endif
    infile >> str; // -fullvocsize
    if(str.compare("-fullvocsize")) {
      cout << "rnnlm-info format incorrect for -fullvocsize";
      exit(1);
    }
    infile >> fullvocsize;
    if(fullvocsize<=0) {
      cout << "rnnlm-info value incorrect for -fullvocsize";
      exit(1);
    }
    infile.close();

    // Reads the language model.
    KaldiRnnlmWrapper rnnlm(opts, unk_prob_rspecifier, word_symbols_rxfilename, rnnlm_rxfilename, true, inputwordlist, outputwordlist, layersizes, fullvocsize, nthread);

    // Reads and writes as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();

      RnnlmDeterministicFst *lm_to_add_orig
               = new RnnlmDeterministicFst(max_ngram_order, &rnnlm, true);
      fst::DeterministicOnDemandFst<StdArc> *lm_to_add =
         new fst::ScaleDeterministicOnDemandFst(lm_scale, lm_to_add_orig);

      KALDI_LOG << key;
      if (acoustic_scale != 1.0) {
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      }
      TopSortCompactLatticeIfNeeded(&clat);
      
      fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(
          &lm_to_subtract_det_scale, lm_to_add);

      // Composes lattice with language model.
      CompactLattice composed_clat;
      ComposeCompactLatticePruned(compose_opts, clat,
                                  &combined_lms, &composed_clat);

      delete lm_to_add_orig;
//      lm_to_add_orig->Clear();

      if (composed_clat.NumStates() == 0) {
        // Something went wrong.  A warning will already have been printed.
        n_fail++;
      } else {
        if (acoustic_scale != 1.0) {
          if (acoustic_scale == 0.0)
            KALDI_ERR << "Acoustic scale cannot be zero.";
          fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                            &composed_clat);
        }
        compact_lattice_writer.Write(key, composed_clat);
        n_done++;
      }
      delete lm_to_add;
      delete lm_to_add_orig;
    }
    delete lm_to_subtract_fst;

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
