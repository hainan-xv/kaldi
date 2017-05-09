// arpa-sampling.cc

#include "arpa-sampling.h"
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <math.h>

namespace kaldi {

/// this function reads each ngram line in the ARPA file
void ArpaSampling::ConsumeNGram(const NGram& ngram) {
  int32 cur_order = ngram.words.size();
  int32 word = ngram.words.back(); // word is the last word in vector words
  HistType history(ngram.words.begin(), ngram.words.begin() + cur_order - 1);
  KALDI_ASSERT(history.size() == cur_order - 1);

  BaseFloat log_prob = ngram.logprob / M_LN10;
  BaseFloat backoff_weight = ngram.backoff / M_LN10;
  std::pair <BaseFloat, BaseFloat> probs_pair;
  probs_pair = std::make_pair(log_prob, backoff_weight);
  // update map
  probs_[cur_order - 1][history].insert({word, probs_pair});
 
  // get vocab_, the map from word string to integer
  const fst::SymbolTable* sym = Symbols();
  if (cur_order == 1) {
    num_words_++;
    std::string word_s = sym->Find(word);
    std::pair<std::string, int32> word_pair;
    word_pair = std::make_pair(word_s, word);
    vocab_.push_back(word_pair);
  }
}

void ArpaSampling::HeaderAvailable() {
  ngram_counts_ = NgramCounts();
  ngram_order_ = NgramCounts().size(); 
  probs_.resize(ngram_order_);
}

// this function returns the probability of the ngram (history, word) for given
// order if the history and the word given the history exists. 
// Otherwise it backoff to previous order to recursively search the lower order
// ngram until backoff to unigram. 
BaseFloat ArpaSampling::GetProb(int32 order, int32 word, const HistType& history) {
  BaseFloat prob = 0.0;
  NgramType::const_iterator it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end() &&
      probs_[order-1][history].find(word) != probs_[order-1][history].end()) {
    prob += probs_[order-1][history][word].first;
  } else { // backoff to the previous order
    order--;
    if (order >= 1) {
      HistType::const_iterator first = history.begin() + 1;
      HistType::const_iterator last = history.end();
      HistType h(first, last);
      prob += GetProb(order, word, h);
      int32 word_new = history.back();
      HistType::const_iterator last_new = history.end() - 1;
      HistType h_new(history.begin(), last_new);
      prob += GetBackoffWeight(order, word_new, h_new);
    }
  }
  return prob;
}

// this function returns the backoff weight of the ngram (history, word)
BaseFloat ArpaSampling::GetBackoffWeight(int32 order, int32 word, const HistType& history) {
  BaseFloat bow = 0.0;
  NgramType::const_iterator it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end()) {
    WordToProbsMap::const_iterator it2 = probs_[order - 1][history].find(word);
    if (it2 != probs_[order - 1][history].end()) {
      bow = it2->second.second;
    }
  }
  return bow;
}

// this function computes the estimated pdf given a history
void ArpaSampling::ComputeWordPdf(const HistType& history, std::vector<std::pair<int32, BaseFloat> >* pdf) {
  int32 order = history.size();
  BaseFloat prob = 0.0;
  (*pdf).resize(num_words_); 
  for (int32 i = 0; i < num_words_; i++) {
    NgramType::const_iterator it = probs_[order].find(history);
    int32 word = vocab_[i].second; 
    if (it != probs_[order].end()) {
      WordToProbsMap::const_iterator it2 = probs_[order][history].find(word);
      if (it2 != probs_[order][history].end()) {
        prob = pow(10, it2->second.first);
        (*pdf)[i].first = word;
        (*pdf)[i].second += prob;
      } else {
        HistType::const_iterator first = history.begin() + 1;
        HistType::const_iterator last = history.end();
        HistType h(first, last);
        int32 word_new = history.back();
        HistType::const_iterator last_new = history.end() - 1;
        HistType h_new(history.begin(), last_new);
        prob = pow(10, GetBackoffWeight(order, word_new, h_new) + GetProb(order, word, h));
        (*pdf)[i].first = word;
        (*pdf)[i].second += prob;
      }
    } else {
      HistType::const_iterator first = history.begin() + 1;
      HistType::const_iterator last = history.end();
      HistType h(first, last);
      int32 word_new = history.back();
      HistType::const_iterator last_new = history.end() - 1;
      HistType h_new(history.begin(), last_new);
      prob = pow(10, GetBackoffWeight(order, word_new, h_new) + GetProb(order, word, h));
      (*pdf)[i].first = word;
      (*pdf)[i].second += prob;
    }
  }
}

// this function computes history weights for given histories
// the total weights of histories is 1
HistWeightsType ArpaSampling::ComputeHistoriesWeights(std::vector<HistType> histories) {
  HistWeightsType hists_weights;
  for (std::vector<HistType>::iterator it = histories.begin(); it != histories.end(); ++it) {
    HistType history(*(it));
    KALDI_ASSERT(history.size() <= ngram_order_);
    for (int32 i = 0; i < history.size() + 1; i++) {
      HistType h_tmp = history;
      BaseFloat prob = 1.0 / histories.size();
      while (h_tmp.size() > (history.size() - i)) {
        HistType::iterator last = h_tmp.end() - 1;
        HistType h(h_tmp.begin(), last);
        int32 word = h_tmp.back();
        prob *= pow(10, GetBackoffWeight(h_tmp.size(), word, h));
        HistType h_up(h_tmp.begin() + 1, h_tmp.end());
        h_tmp = h_up;
      }
      HistType::iterator begin = history.begin() + i;
      HistType h(begin, history.end());
      hists_weights[h] += prob;
    }
  }
  return hists_weights;
}

// Get weighted pdf given a list of histories
void ArpaSampling::ComputeWeightedPdf(HistWeightsType hists_weights, 
    std::vector<std::pair<int32, BaseFloat> >* pdf_w) {
  BaseFloat prob = 0;
  (*pdf_w).clear();
  (*pdf_w).resize(num_words_);
  for (int32 i = 0; i < num_words_; i++) {
    for (HistWeightsType::const_iterator it = hists_weights.begin(); 
        it != hists_weights.end(); ++it) {
      HistType h(it->first);
      int32 order = h.size();
      NgramType::const_iterator it_hist = probs_[order].find(h);
      if (it_hist != probs_[order].end()) {
        int32 word = vocab_[i].second;
        WordToProbsMap::const_iterator it_word = probs_[order][h].find(word);
        if (it_word != probs_[order][h].end()) {
          if (order > 0) {
            HistType::iterator last = h.end() - 1;
            HistType::iterator first = h.begin() + 1;
            HistType h1(h.begin(), last);
            HistType h2(first, h.end());
            prob = it->second * (pow(10, probs_[order][h][word].first) - 
                    pow(10, GetBackoffWeight(order, h.back(), h1) + GetProb(order, word, h2)));
            (*pdf_w)[i].first = word;
            (*pdf_w)[i].second += prob;
          } else {
            prob = it->second * pow(10, probs_[order][h][word].first);
            (*pdf_w)[i].first = word;
            (*pdf_w)[i].second += prob;
          }
        }
      }
    } // end reading history
  } // end reading words
}

// this function compute words existing for given histories and their corresponding
// probabilities
void ArpaSampling::ComputeOutputWords(std::vector<HistType> histories,
    unordered_map<int32, BaseFloat>* pdf_w) {
  HistWeightsType hists_weights = ComputeHistoriesWeights(histories); 
  BaseFloat prob = 0;
  for (HistWeightsType::const_iterator it = hists_weights.begin(); it != hists_weights.end(); ++it) {
    HistType h(it->first);
    int32 order = h.size();
    NgramType::const_iterator it_hist = probs_[order].find(h);
    if (it_hist != probs_[order].end()) {
      for(WordToProbsMap::const_iterator it_word = probs_[order][h].begin(); 
          it_word != probs_[order][h].end(); ++it_word) {
        int32 word = it_word->first;
        if (order > 0) {
          HistType::iterator last = h.end() - 1;
          HistType::iterator first = h.begin() + 1;
          HistType h1(h.begin(), last);
          HistType h2(first, h.end());
          prob = it->second * (pow(10, probs_[order][h][word].first) - 
                  pow(10, GetBackoffWeight(order, h.back(), h1) + GetProb(order, word, h2)));
          unordered_map<int32, BaseFloat>::iterator map_it = (*pdf_w).find(word);
          if (map_it != (*pdf_w).end()) {
            (*pdf_w)[word] += prob;
          } else {
            (*pdf_w).insert({word, prob});
          }
        }
      }
    }
  }
}

// this function randomly generate 5 - 1005 histories 
std::vector<HistType> ArpaSampling::RandomGenerateHistories() {
  std::vector<HistType> histories;
  int32 num_histories = rand() % 1000 + 5; // generate at least 5 histories
  for (int32 i = 0; i < num_histories; i++) {
    HistType hist;
    // size of history should be in {1, 2, ..., ngram_order_}
    int32 size_hist = rand() % (ngram_order_ - 1) + 1;
    KALDI_ASSERT(size_hist <= ngram_order_);
    for (int32 j = 0; j < size_hist; j++) {
      // word can not be zero since zero represents epsilon in the fst symbol format
      int32 word = rand() % (vocab_.size() - 1) + 1;
      KALDI_ASSERT(word > 0 && word <= vocab_.size());
      hist.push_back(word);
    }
    histories.push_back(hist);
  }
  return histories;
}

// this function checks the two estimated pdfs from 1) weighted history 
// and 2) normal computation are the same
void ArpaSampling::TestPdfsEqual() {
  std::vector<HistType> histories;
  histories = RandomGenerateHistories();
  HistWeightsType hists_weights;
  hists_weights = ComputeHistoriesWeights(histories);
  std::vector<std::pair<int32, BaseFloat> > pdf_hist_weight;
  ComputeWeightedPdf(hists_weights, &pdf_hist_weight);
  // check the averaged pdf sums to 1
  BaseFloat sum = 0;
  for (int32 i = 0; i < num_words_; i++) {
    sum += pdf_hist_weight[i].second;
  }
  KALDI_ASSERT(ApproxEqual(sum, 1.0));
  // get the average pdf
  std::vector<std::pair<int32, BaseFloat> > pdf;
  pdf.resize(num_words_);
  for (int32 i = 0; i < histories.size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > pdf_h;
    ComputeWordPdf(histories[i], &pdf_h);
    for(int32 j = 0; j < pdf_h.size(); j++) {
      pdf[j].first = pdf_h[j].first;
      pdf[j].second += pdf_h[j].second / histories.size();
    }
  }
  // check the averaged pdf sums to 1
  sum = 0;
  for (int32 i = 0; i < num_words_; i++) {
    sum += pdf[i].second;
  }
  KALDI_ASSERT(ApproxEqual(sum, 1.0));
  // check equality of the two pdfs
  BaseFloat diff = 0;
  for (int32 i = 0; i < num_words_; i++) {
    diff += abs(pdf_hist_weight[i].second - pdf[i].second);
  }
  KALDI_ASSERT(ApproxEqual(diff, 0.0));
}

// Test the read-in language model
void ArpaSampling::TestReadingModel() {
  KALDI_LOG << "Testing model reading part..."<< std::endl;
  KALDI_LOG << "Vocab size is: " << vocab_.size();
  KALDI_LOG << "Ngram_order is: " << ngram_order_;
  KALDI_ASSERT(probs_.size() == ngram_counts_.size());
  for (int32 i = 0; i < ngram_order_; i++) {
    int32 size_ngrams = 0;
    KALDI_LOG << "Test: for order " << (i + 1);
    KALDI_LOG << "Expected number of " << (i + 1) << "-grams: " << ngram_counts_[i];
    for (NgramType::const_iterator it1 = probs_[i].begin(); it1 != probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (WordToProbsMap::const_iterator it2 = probs_[i][h].begin(); it2 != probs_[i][h].end(); ++it2) {
        size_ngrams++; // number of words given
      }
    }
    KALDI_LOG << "Read in number of " << (i + 1) << "-grams: " << size_ngrams;
  }
  KALDI_LOG << "Assert sum of unigram probs equal to 1...";
  BaseFloat prob_sum = 0.0;
  int32 count = 0;
  for (NgramType::const_iterator it1 = probs_[0].begin(); it1 != probs_[0].end();++it1) {
    HistType h(it1->first);
    for (WordToProbsMap::const_iterator it2 = probs_[0][h].begin(); it2 != probs_[0][h].end(); ++it2) {
      prob_sum += 1.0 * pow(10.0, it2->second.first);
      count++;
    }
  }
  KALDI_LOG << "Number of total words: " << count;
  KALDI_LOG << "Sum of unigram probs equal to " << prob_sum;

  KALDI_LOG << "Assert sum of bigram probs given a history equal to 1...";
  prob_sum = 0.0;
  NgramType::const_iterator it1 = probs_[1].begin();
  HistType h(it1->first);
  for (int32 i = 0; i < num_words_; i++) {
    WordToProbsMap::const_iterator it2 = probs_[1][h].find(vocab_[i].second);
    if (it2 != probs_[1][h].end()) {
      prob_sum += 1.0 * pow(10, it2->second.first);
    } else {
      prob_sum += pow(10, GetProb(2, vocab_[i].second, h));
    }
  }
  KALDI_LOG << "Sum of bigram probs given a history equal to " << prob_sum;
}

int32 ArpaSampling::GetNgramOrder() {
  return ngram_order_;
}

// Read histories of integers from a file
std::vector<HistType> ArpaSampling::ReadHistories(std::istream &is, bool binary) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = Symbols();
  std::vector<HistType> histories;
  std::string line;
  KALDI_LOG << "Start reading histories from file...";
  while (getline(is, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    HistType history;
    int32 word;
    for (int32 i = 0; i < tokens.size(); i++) {
      word = sym->Find(tokens[i]);
      if (word == fst::SymbolTable::kNoSymbol) {
        word = sym->Find(unk_symbol_);
      }
      history.push_back(word);
    }
    if (history.size() >= ngram_order_) {
      HistType h(history.end() - ngram_order_ + 1, history.end());
      history.clear();
      HistType history = h;
    }
    histories.push_back(history);
  }
  KALDI_LOG << "Finished reading histories from file.";
  return histories;
}

} // end of kaldi
