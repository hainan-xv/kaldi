// lm/kaldi-rnnlm.cc

// Copyright 2015  Guoguo Chen
//	     2016  Ricky Chan Ho Yin

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

#include <utility>

#include "lm/kaldi-rnnlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

KaldiRnnlmWrapper::~KaldiRnnlmWrapper() {
  if(use_cued_lm && cuedrnnlm_ptr_) {
    delete cuedrnnlm_ptr_;
    cuedrnnlm_ptr_ = NULL;
  }
}

KaldiRnnlmWrapper::KaldiRnnlmWrapper(
    const KaldiRnnlmWrapperOpts &opts,
    const std::string &unk_prob_rspecifier,
    const std::string &word_symbol_table_rxfilename,
    const std::string &rnnlm_rxfilename) {
  rnnlm_.setRnnLMFile(rnnlm_rxfilename);
  rnnlm_.setRandSeed(1);
  rnnlm_.setUnkSym(opts.unk_symbol);
  rnnlm_.setUnkPenalty(unk_prob_rspecifier);
  rnnlm_.restoreNet();

  // Reads symbol table.
  fst::SymbolTable *word_symbols = NULL;
  if (!(word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }
  label_to_word_.resize(word_symbols->NumSymbols() + 1);
  for (int32 i = 0; i < label_to_word_.size() - 1; ++i) {
    label_to_word_[i] = word_symbols->Find(i);
    if (label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }
  label_to_word_[label_to_word_.size() - 1] = opts.eos_symbol;
  eos_ = label_to_word_.size() - 1;
  use_cued_lm = false;
}

KaldiRnnlmWrapper::KaldiRnnlmWrapper(
    const KaldiRnnlmWrapperOpts &opts,
    const std::string &unk_prob_rspecifier,
    const std::string &word_symbol_table_rxfilename,
    const std::string &rnnlm_rxfilename,
    bool use_cued,
    const std::string &inputwlist,
    const std::string &outputwlist,
    std::vector<int> &lsizes,
    int fvocsize,
    int nthread /* =1 */) {

    use_cued_lm = use_cued;
    if(!use_cued_lm) {
        rnnlm_.setRnnLMFile(rnnlm_rxfilename);
        rnnlm_.setRandSeed(1);
        rnnlm_.setUnkSym(opts.unk_symbol);
        rnnlm_.setUnkPenalty(unk_prob_rspecifier);
        rnnlm_.restoreNet();
    }
    else {
        cuedrnnlm_ptr_ = new RNNLM(lsizes, fvocsize);
        // text rnn model, minibatch 1 for evaluation, debug 0
        cout << "full size is " << cuedrnnlm_ptr_->fullvocsize << endl;
        cuedrnnlm_ptr_->loadfresh(rnnlm_rxfilename, inputwlist, outputwlist, false, 1, 0);
        /* cuedrnnlm_ptr_->loadfresh(rnnlm_rxfilename, inputwlist, outputwlist, false, 1, 0, 0, false); */
        /* cuedrnnlm_ptr_->setLognormConst(-1.0);  // default lognormconst value: -1.0 in cued rnnlm
           cuedrnnlm_ptr_->setNthread(nthread);          // default nthread value: 1 in cued rnnlm */
        lsizes = cuedrnnlm_ptr_->layersizes;
        cuedrnnlm_ptr_->fullvocsize = fvocsize;
        cout << "full size is " << cuedrnnlm_ptr_->fullvocsize << endl;
    }

    // Reads symbol table.
    fst::SymbolTable *word_symbols = NULL;
    if (!(word_symbols =
          fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
        KALDI_ERR << "Could not read symbol table from file "
            << word_symbol_table_rxfilename;
    }
    label_to_word_.resize(word_symbols->NumSymbols() + 1);
    for (int32 i = 0; i < label_to_word_.size() - 1; ++i) {
        label_to_word_[i] = word_symbols->Find(i);
        if (label_to_word_[i] == "") {
            KALDI_ERR << "Could not find word for integer " << i << "in the word "
                << "symbol table, mismatched symbol table or you have discoutinuous "
                << "integers in your symbol table?";
        }
    }
    label_to_word_[label_to_word_.size() - 1] = opts.eos_symbol;
    eos_ = label_to_word_.size() - 1;
    delete word_symbols;
}

BaseFloat KaldiRnnlmWrapper::GetLogProb(
    int32 word, const std::vector<int32> &wseq,
    const std::vector<float> &context_in,
    std::vector<float> *context_out) {


  std::vector<std::string> wseq_symbols(wseq.size());
  for (int32 i = 0; i < wseq_symbols.size(); ++i) {
    KALDI_ASSERT(wseq[i] < label_to_word_.size());
    wseq_symbols[i] = label_to_word_[wseq[i]];
  }

  if(use_cued_lm)
    return cuedrnnlm_ptr_->computeConditionalLogprob(label_to_word_[word], wseq_symbols,
                                          context_in, context_out);
  else
    return rnnlm_.computeConditionalLogprob(label_to_word_[word], wseq_symbols,
                                          context_in, context_out);
}

void KaldiRnnlmWrapper::ResetCuedLMhist() {
  cuedrnnlm_ptr_->ResetRechist();
}

int32 KaldiRnnlmWrapper::GetHiddenLayerSize() {
  if(use_cued_lm==false)
    return rnnlm_.getHiddenLayerSize();
  else
    return cuedrnnlm_ptr_->getOneHiddenLayerSize();
}

RnnlmDeterministicFst::RnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiRnnlmWrapper *rnnlm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;

  // Uses empty history for <s>.
  std::vector<Label> bos;
  std::vector<float> bos_context(rnnlm->GetHiddenLayerSize(), 1.0);
  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(bos_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
  use_cued_lm = false;
}

RnnlmDeterministicFst::RnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiRnnlmWrapper *rnnlm, bool using_cued_lm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;

  use_cued_lm = using_cued_lm;
  if(use_cued_lm) {
    rnnlm->ResetCuedLMhist();
  }
  float init_value = (use_cued_lm == true) ? CUED_RESETVALUE : 1.0;

  // Uses empty history for <s>.
  std::vector<Label> bos;
  std::vector<float> bos_context(rnnlm->GetHiddenLayerSize(), init_value);
  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(bos_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight RnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  cout << "computing final " << endl;
  BaseFloat logprob = rnnlm_->GetLogProb(rnnlm_->GetEos(), wseq,
                                         state_to_context_[s], NULL);
  return Weight(-logprob);
}

bool RnnlmDeterministicFst::GetArc(StateId s, Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  std::vector<float> new_context(rnnlm_->GetHiddenLayerSize());
  BaseFloat logprob = rnnlm_->GetLogProb(ilabel, wseq,
                                         state_to_context_[s], &new_context);
  cout << "new weight is " << logprob << endl;

  wseq.push_back(ilabel);
  if (max_ngram_order_ > 0) {
    while (wseq.size() >= max_ngram_order_) {
      // History state has at most <max_ngram_order_> - 1 words in the state.
      wseq.erase(wseq.begin(), wseq.begin() + 1);
    }
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      wseq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <lseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to <state_to_wseq_> and
  // <state_to_context_>.
  if (result.second == true) {
    state_to_wseq_.push_back(wseq);
    state_to_context_.push_back(new_context);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  cout << "new weight is " << oarc->weight << endl;
  return true;
}

}  // namespace kaldi
