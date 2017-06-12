// lm/kaldi-rnnlm.cc

#include <utility>
#include <fstream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "tensorflow/tensorflow-rnnlm-lib.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

using tensorflow::Status;

namespace kaldi {
using tf_rnnlm::KaldiTfRnnlmWrapper;
using tf_rnnlm::TfRnnlmDeterministicFst;
using std::ifstream;

KaldiTfRnnlmWrapper::KaldiTfRnnlmWrapper(
    const KaldiTfRnnlmWrapperOpts &opts,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename, // TODO(hxu) will do this later
    const std::string &unk_prob_rspecifier,
//    Session* session) {
    const std::string &tf_model_path) {
//  session_ = session;
  {
    string graph_path = tf_model_path + "/meta";

    Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session_);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

    tensorflow::MetaGraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

    // Add the graph to the session
    status = session_->Create(graph_def.graph_def());
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

    Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = tf_model_path;
    
    status = session_->Run(
              {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
              {},
              {graph_def.saver_def().restore_op_name()},
              nullptr);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

  }

  fst::SymbolTable *fst_word_symbols = NULL;
  if (!(fst_word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }

  fst_label_to_word_.resize(fst_word_symbols->NumSymbols());

  for (int32 i = 0; i < fst_label_to_word_.size(); ++i) {
    fst_label_to_word_[i] = fst_word_symbols->Find(i);
    if (fst_label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }

  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);

  { // input.
    ifstream ifile(rnn_wordlist.c_str());
    int id;
    string word;
    int i = 0;
    while (ifile >> id >> word) { // TODO(hxu) ugly fix for cued-rnnlm's bug
                                  // will implement a better fix later
      if (word == "[UNK]") {
        word = "<unk>";
      } else if (word == "<OOS>") {
        continue;
      }
      i++;
      assert(i == id + 1);
      rnn_label_to_word_.push_back(word);

      int fst_label = fst_word_symbols->Find(rnn_label_to_word_[i]);
      KALDI_ASSERT(fst::SymbolTable::kNoSymbol != fst_label);
      fst_label_to_rnn_label_[fst_label] = i;
    }
    bos_ = 1;
    eos_ = 0; // TODO(hxu)
  }
  rnn_label_to_word_.push_back("<OOS>");
  
  for (int i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = rnn_label_to_word_.size() - 1;
    }
  }


}

BaseFloat KaldiTfRnnlmWrapper::GetLogProb(
    int32 word, const std::vector<int32> &wseq,
    const Tensor &context_in,
    tensorflow::Tensor *context_out) {

  std::vector<std::string> wseq_symbols(wseq.size());
  for (int32 i = 0; i < wseq_symbols.size(); ++i) {
    KALDI_ASSERT(wseq[i] < label_to_word_.size());
    wseq_symbols[i] = label_to_word_[wseq[i]];
  }

  std::vector<std::pair<string, Tensor>> inputs;

  Tensor lastword(tensorflow::DT_INT32, {1, 1});
  Tensor thisword(tensorflow::DT_INT32, {1, 1});

  lastword.scalar<int32>()() = (wseq.size() == 0? bos_: wseq.back());
  thisword.scalar<int32>()() = word;

  inputs = {
    {"Train/Model/test_word_in", lastword},
    {"Train/Model/test_word_out", thisword},
    {"Train/Model/test_state", context_in},
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  Status status = session_->Run(inputs, {"Train/Model/test_out", "Train/Model/test_state_out"}, {}, &outputs);

//  return rnnlm_.computeConditionalLogprob(label_to_word_[word], wseq_symbols,
//                                          context_in, context_out);
  if (context_out != NULL)
    *context_out = outputs[1];
  return outputs[0].scalar<float>()();
}

void KaldiTfRnnlmWrapper::GetInitialContext(Tensor *c) const {
  std::vector<Tensor> state;
  Status status = session_->Run(std::vector<std::pair<string, tensorflow::Tensor>>(), {"Train/Model/test_initial_state"}, {}, &state);
  *c = state[0];
}

TfRnnlmDeterministicFst::TfRnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiTfRnnlmWrapper *rnnlm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;

  // Uses empty history for <s>.
  std::vector<Label> bos;
//  std::vector<float> bos_context(rnnlm->GetHiddenLayerSize(), 1.0);

  Tensor initial_context;
  rnnlm_->GetInitialContext(&initial_context);

  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(initial_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight TfRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = rnnlm_->GetLogProb(rnnlm_->GetEos(), wseq,
                                         state_to_context_[s], NULL);
  return Weight(-logprob);
}

bool TfRnnlmDeterministicFst::GetArc(StateId s, Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  tensorflow::Tensor new_context;

  int32 rnn_word = rnnlm_->fst_label_to_rnn_label_[ilabel];
  BaseFloat logprob = rnnlm_->GetLogProb(rnn_word, wseq,
                                         state_to_context_[s], &new_context);

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

  return true;
}

}  // namespace kaldi
