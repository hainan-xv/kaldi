// rnnlm/kaldi-rnnlm-rescoring.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//                Yiming Wang
//
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

#include "rnnlm/kaldi-rnnlm-rescoring.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet3 {

void KaldiRnnlmDeterministicFst::ReadFstWordSymbolTableAndRnnWordlist(
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename) {
  // Reads symbol table.
  fst::SymbolTable *fst_word_symbols = NULL;
  if (!(fst_word_symbols =
      fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
              << word_symbol_table_rxfilename;
  }

  full_voc_size_ = fst_word_symbols->NumSymbols();
  fst_label_to_word_.resize(full_voc_size_);

  for (int32 i = 0; i < fst_label_to_word_.size(); ++i) {
    fst_label_to_word_[i] = fst_word_symbols->Find(i);
    if (fst_label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
                << "symbol table, mismatched symbol table or you have discoutinuous "
                << "integers in your symbol table?";
    }
  }

  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);

  rnn_label_to_word_.push_back("<s>");
  rnn_label_to_word_.push_back("<OOS>");
  out_OOS_index_ = 1;
  { // input
    std::ifstream ifile(rnn_wordlist.c_str());
    int32 id;
    string word;
    int32 i = 1;
    while (ifile >> id >> word) { // TODO(hxu) ugly fix for cued-rnnlm's bug
                                  // will implement a better fix later
      if (word == "[UNK]") {
        word = "<unk>";
      } else if (word == "<OOS>") {
        continue;
      }
      i++;
      assert(i == id + 2);
      rnn_label_to_word_.push_back(word);

      int fst_label = fst_word_symbols->Find(rnn_label_to_word_[i]);
      KALDI_ASSERT(fst::SymbolTable::kNoSymbol != fst_label);
      fst_label_to_rnn_label_[fst_label] = i;
    }
  }

  for (int32 i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = out_OOS_index_;
    }
  }
}

KaldiRnnlmDeterministicFst::KaldiRnnlmDeterministicFst(int32 max_ngram_order,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename,
    const rnnlm::LmNnet &lm_nnet) {
  max_ngram_order_ = max_ngram_order;
  ReadFstWordSymbolTableAndRnnWordlist(rnn_wordlist,
                                       word_symbol_table_rxfilename);

  std::vector<Label> bos;
  bos.push_back(0); // 0 for <s>
  state_to_wseq_.push_back(bos);
  DecodableRnnlmSimpleLoopedComputationOptions opts;
  DecodableRnnlmSimpleLoopedInfo info(opts, lm_nnet);
  DecodableRnnlmSimpleLooped decodable_rnnlm(info);
  state_to_decodable_rnnlm_.push_back(decodable_rnnlm);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight KaldiRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  const std::vector<Label>& wseq = state_to_wseq_[s];
  state_to_decodable_rnnlm_[s].TakeFeatures(std::vector<Label>(1, wseq.back()));
  // log prob of end of sentence
  BaseFloat logprob = state_to_decodable_rnnlm_[s].GetOutput(0, 0);
  return Weight(-logprob);
}

bool KaldiRnnlmDeterministicFst::GetArc(StateId s, Label ilabel,
                                        fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  DecodableRnnlmSimpleLooped decodable_rnnlm = state_to_decodable_rnnlm_[s];
  decodable_rnnlm.TakeFeatures(std::vector<Label>(1, wseq.back()));
  int32 rnn_word = fst_label_to_rnn_label_[ilabel];
  BaseFloat logprob = decodable_rnnlm.GetOutput(0, rnn_word);
  if (rnn_word == out_OOS_index_)
    logprob = logprob - Log(full_voc_size_ - rnn_label_to_word_.size() + 1.0);

  wseq.push_back(rnn_word);
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
  // <state_to_decodable_rnnlm_>.
  if (result.second == true) {
    state_to_wseq_.push_back(wseq);
    state_to_decodable_rnnlm_.push_back(decodable_rnnlm);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace nnet3
}  // namespace kaldi
