// arpa-sampling.cc

#include "arpa-sampling.h"
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <math.h>

namespace kaldi {

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

BaseFloat ArpaSampling::GetProb(int32 order, int32 word, const HistType& history) {
  BaseFloat prob = 0.0;
  auto it = probs_[order - 1].find(history);
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

BaseFloat ArpaSampling::GetBackoffWeight(int32 order, int32 word, const HistType& history) {
  BaseFloat bow = 0.0;
  auto it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end()) {
    auto it2 = probs_[order - 1][history].find(word);
    if (it2 != probs_[order - 1][history].end()) {
      bow = (it2->second).second;
    }
  }
  return bow;
}

void ArpaSampling::ComputeWordPdf(const HistType& history, std::vector<std::pair<int32, BaseFloat> >* pdf) {
  int32 order = history.size();
  BaseFloat prob = 0.0;
  (*pdf).resize(num_words_); // if do not do this, (*pdf)[word] += prob will get seg fault
  for (int32 i = 0; i < num_words_; i++) {
    auto it = probs_[order].find(history);
    int32 word = vocab_[i].second; // get word from the map
    if (it != probs_[order].end()) {
      auto it2 = probs_[order][history].find(word);
      if (it2 != probs_[order][history].end()) {
        prob = pow(10, (it2->second).first);
        (*pdf)[i].first = word;
        (*pdf)[i].second += prob;
      } else {
        HistType::const_iterator first = history.begin() + 1;
        HistType::const_iterator last = history.end();
        HistType h(first, last);
        int32 word_new = history.back();
        HistType::const_iterator last_new = history.end() - 1;
        HistType h_new(history.begin(), last_new);
        prob = pow(10, GetBackoffWeight(order, word_new, h_new)) *
 
          pow(10, GetProb(order, word, h));
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
      prob = pow(10, GetBackoffWeight(order, word_new, h_new)) *
             pow(10, GetProb(order, word, h));
      (*pdf)[i].first = word;
      (*pdf)[i].second += prob;
    }
  }
}

// Get history weights
void ArpaSampling::ComputeHistoriesWeights() {
  for (auto it = histories_.begin(); it != histories_.end(); ++it) {
    HistType history(*(it));
    KALDI_ASSERT(history.size() <= ngram_order_);
    for (int32 i = 0; i < history.size() + 1; i++) {
      HistType h_tmp = history;
      BaseFloat prob = 1.0 / histories_.size();
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
      hists_weights_[h] += prob;
    }
  }
}

// Get weighted pdf
void ArpaSampling::ComputeWeightedPdf(std::vector<std::pair<int32, BaseFloat> >* pdf_w) {
  BaseFloat prob = 0;
  (*pdf_w).clear();
  (*pdf_w).resize(num_words_); // if do not do this, (*pdf_w)[word] += prob will get seg fault
  for (int32 i = 0; i < num_words_; i++) {
    for (auto it = hists_weights_.begin(); it != hists_weights_.end(); ++it) {
      HistType h(it->first);
      int32 order = h.size();
      auto it_hist = probs_[order].find(h);
      if (it_hist != probs_[order].end()) {
        int32 word = vocab_[i].second;
        auto it_word = probs_[order][h].find(word);
        if (it_word != probs_[order][h].end()) {
          if (order > 0) {
            HistType::iterator last = h.end() - 1;
            HistType::iterator first = h.begin() + 1;
            HistType h1(h.begin(), last);
            HistType h2(first, h.end());
            prob = (it->second) * (pow(10, probs_[order][h][word].first) - 
                    pow(10, GetBackoffWeight(order, h.back(), h1))
                    * pow(10, GetProb(order, word, h2)));
            (*pdf_w)[i].first = word;
            (*pdf_w)[i].second += prob;
          } else {
            prob = (it->second) * pow(10, probs_[order][h][word].first);
            (*pdf_w)[i].first = word;
            (*pdf_w)[i].second += prob;
          }
        }
      }
    } // end reading history
  } // end reading words
}

void ArpaSampling::RandomGenerateHistories() {
  // clear previous histories
  histories_.clear();
  // randomly generate histories
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
    histories_.push_back(hist);
  }
}

// this function checks the two estimated pdfs from 1) weighted history 
// and 2) normal computation are the same
void ArpaSampling::TestPdfsEqual() {
  RandomGenerateHistories();
  hists_weights_.clear();
  ComputeHistoriesWeights();
  std::vector<std::pair<int32, BaseFloat> > pdf_hist_weight;
  ComputeWeightedPdf(&pdf_hist_weight);
  // check the averaged pdf sums to 1
  BaseFloat sum = 0;
  for (int32 i = 0; i < num_words_; i++) {
    sum += pdf_hist_weight[i].second;
  }
  KALDI_ASSERT(ApproxEqual(sum, 1.0));
  // get the average pdf
  std::vector<std::pair<int32, BaseFloat> > pdf;
  pdf.resize(num_words_);
  for (int32 i = 0; i < histories_.size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > pdf_h;
    ComputeWordPdf(histories_[i], &pdf_h);
    for(int32 j = 0; j < pdf_h.size(); j++) {
      pdf[j].first = pdf_h[j].first;
      pdf[j].second += pdf_h[j].second / histories_.size();
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

// this function returns the log probability of the given sentence
BaseFloat ArpaSampling::ComputeSentenceProb(const std::vector<int32>& sentence) {
  BaseFloat prob = 0;
  const fst::SymbolTable* sym = Symbols();
  for (int32 i = 1; i < sentence.size(); i++) {
    if (i < ngram_order_ - 1) {
      HistType::const_iterator last = sentence.begin() + i;
      HistType h(sentence.begin(), last);
      prob += GetProb(i + 1, sentence[i], h);
    } else {
      HistType::const_iterator first = sentence.begin() + i + 1 - ngram_order_;
      HistType::const_iterator last = sentence.begin() + i;
      HistType h(first, last);
      KALDI_ASSERT(h.size() == ngram_order_ - 1);
      prob += GetProb(ngram_order_, sentence[i], h);
    }
    std::string word_s = sym->Find(sentence[i]);
    if (sentence[i] == kUnk) {
      word_s = unk_symbol_;
    }
  }
  return prob;
}

// this functions computes the total log probability of all test sentences
BaseFloat ArpaSampling::ComputeAllSentencesProb(const std::vector<std::vector<int32> >& sentences) {
  BaseFloat prob = 0;
  for (int32 i = 0; i < sentences.size(); i++) {
    KALDI_ASSERT(sentences[i].size() >= 3);
    prob += ComputeSentenceProb(sentences[i]);
  }
  int32 len = sentences.size();
  KALDI_LOG << "Total log-probabilities of " << len << " sentences are: "\
    << prob;
  return prob;
}

void ArpaSampling::PrintHist(const HistType& h) {
  KALDI_LOG << "Current hist is: ";
  for (int32 i = 0; i < h.size(); i++) {
    KALDI_LOG << h[i] << " ";
  }
}

// Test the read-in model by computing the total prob of given sentences
void ArpaSampling::TestProbs(std::istream &is, bool binary) {
  std::vector<std::vector<int32> > sentences;
  ReadSentences(is, &sentences);
  ComputeAllSentencesProb(sentences);
}

// Test the read-in language model
void ArpaSampling::TestReadingModel() {
  KALDI_LOG << "Testing model reading part..."<< std::endl;
  KALDI_LOG << "Vocab size is: " << vocab_.size();
  std::cout << "Print out vocab: " << std::endl;
  for (int i = 0; i < vocab_.size(); i++) {
    std::cout << i << " , " << vocab_[i].first << " , " << vocab_[i].second << std::endl;
  }
  KALDI_LOG << "Ngram_order is: " << ngram_order_;
  KALDI_ASSERT(probs_.size() == ngram_counts_.size());
  for (int32 i = 0; i < ngram_order_; i++) {
    int32 size_ngrams = 0;
    KALDI_LOG << "Test: for order " << (i + 1);
    KALDI_LOG << "Expected number of " << (i + 1) << "-grams: " << ngram_counts_[i];
    for (auto it1 = probs_[i].begin(); it1 != probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (auto it2 = (probs_[i])[h].begin(); it2 != (probs_[i])[h].end(); ++it2) {
        size_ngrams++; // number of words given
      }
    }
    KALDI_LOG << "Read in number of " << (i + 1) << "-grams: " << size_ngrams;
  }
  KALDI_LOG << "Assert sum of unigram probs equal to 1...";
  BaseFloat prob_sum = 0.0;
  int32 count = 0;
  for (auto it1 = (probs_[0]).begin(); it1 != (probs_[0]).end();++it1) {
    HistType h(it1->first);
    for (auto it2 = (probs_[0])[h].begin(); it2 != (probs_[0])[h].end(); ++it2) {
      prob_sum += 1.0 * pow(10.0, (it2->second).first);
      count++;
    }
  }
  KALDI_LOG << "Number of total words: " << count;
  KALDI_LOG << "Sum of unigram probs equal to " << prob_sum;

  KALDI_LOG << "Assert sum of bigram probs given a history equal to 1...";
  prob_sum = 0.0;
  auto it1 = probs_[1].begin();
  HistType h(it1->first);
  for (int32 i = 0; i < num_words_; i++) {
    auto it2 = probs_[1][h].find(vocab_[i].second);
    if (it2 != probs_[1][h].end()) {
      prob_sum += 1.0 * pow(10, (it2->second).first);
    } else {
      prob_sum += pow(10, GetProb(2, vocab_[i].second, h));
    }
  }
  KALDI_LOG << "Sum of bigram probs given a history equal to " << prob_sum;
}

// Read sentences from a file
void ArpaSampling::ReadSentences(std::istream &iss, std::vector<std::vector<int32> >* sentences) {
  const fst::SymbolTable* sym = Symbols();
  std::string line;
  KALDI_LOG << "Start reading sentences...";
  while (getline(iss, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    std::vector<int32> sentence;
    int32 word;
    int32 bos = sym->Find(bos_symbol_);
    sentence.push_back(bos);
    for (int32 i = 0; i < tokens.size(); i++) {
      word = sym->Find(tokens[i]);
      if (word == fst::SymbolTable::kNoSymbol) {
        word = sym->Find(unk_symbol_);
      }
      sentence.push_back(word);
    }
    int32 eos = sym->Find(eos_symbol_);
    sentence.push_back(eos);
    (*sentences).push_back(sentence);
  }
  KALDI_LOG << "Finished reading sentences.";
}

// Read histories of integers from a file
void ArpaSampling::ReadHistories(std::istream &is, bool binary) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = Symbols();
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
      std::reverse(history.begin(), history.end());
      history.resize(ngram_order_ - 1);
      std::reverse(history.begin(), history.end());
    }
    histories_.push_back(history);
  }
  KALDI_LOG << "Finished reading histories from file.";
}

} // end of kaldi
