// Copyright 2017 Hainan Xu
// wrapper for tensorflow rnnlm

#ifndef KALDI_LM_TENSORFLOW_LIB_H_
#define KALDI_LM_TENSORFLOW_LIB_H_

#include <string>
#include <vector>
#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"

#include "tensorflow/core/public/session.h"

using tensorflow::Session;
using tensorflow::Tensor;

namespace kaldi {
namespace tf_rnnlm {

struct KaldiTfRnnlmWrapperOpts {
  std::string unk_symbol;
  std::string bos_symbol;
  std::string eos_symbol;

  KaldiTfRnnlmWrapperOpts() : unk_symbol("<RNN_UNK>"), bos_symbol("<s>"), eos_symbol("</s>") {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in rnnlm.");
    opts->Register("bos-symbol", &eos_symbol, "Beginning of setence symbol in "
                   "rnnlm.");
    opts->Register("eos-symbol", &eos_symbol, "End of setence symbol in "
                   "rnnlm.");
  }
};

class KaldiTfRnnlmWrapper {
 public:
  KaldiTfRnnlmWrapper(const KaldiTfRnnlmWrapperOpts &opts,
                    const std::string &rnn_wordlist,
                    const std::string &word_symbol_table_rxfilename,
                    const std::string &unk_prob_rspecifier,
                    Session* session);

  int32 GetEos() const { return eos_; }
  int32 GetBos() const { return bos_; }
  void GetInitialContext(Tensor* context) const;

  BaseFloat GetLogProb(int32 word, const std::vector<int32> &wseq,
                       const Tensor &context_in,
                       Tensor *context_out);

  std::vector<int> fst_label_to_rnn_label_;
  std::vector<std::string> rnn_label_to_word_;
  std::vector<std::string> fst_label_to_word_;
 private:

  Session* session_;  // ptf not owned here
  std::vector<std::string> label_to_word_;
  int32 eos_;
  int32 bos_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiTfRnnlmWrapper);
};

class TfRnnlmDeterministicFst
    : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  TfRnnlmDeterministicFst(int32 max_ngram_order, KaldiTfRnnlmWrapper *rnnlm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  KaldiTfRnnlmWrapper *rnnlm_;
  int32 max_ngram_order_;
  std::vector<tensorflow::Tensor> state_to_context_;
};

}  // namespace tf_rnnlm
}  // namespace kaldi

#endif  // KALDI_LM_MIKOLOV_RNNLM_LIB_H_
