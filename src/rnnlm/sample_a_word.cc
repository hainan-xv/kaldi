// sample_a_word.cc

#include "sample_a_word.h"
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <math.h>

// Constructor for sampling the next word
NgramModel::NgramModel(char* arpa_file, char* histories_file) {
  vocab_size_ = 0;
  ReadARPAModel(arpa_file);
  ReadHistories(histories_file);
}

// Read language model from a ARPA-format file.
void NgramModel::ReadARPAModel(char* file) {
  std::ifstream data_input(file);
  if (!data_input.is_open()) {
    std::cerr << "error opening '" << file
              << "' for reading\n";
    exit(1);
  }
  std::string line;
  int32 order;
  int32 order_current = 0;
  int32 word;
  int32 iter = 0;
  int32 while_iter = 0;
  std::pair <float, float> probs_pair;
  float log_prob;
  float backoff_weight;
  bool unigram_check = false;
  std::cout << "Start reading ARPA-format file..." << std::endl;
  while (getline(data_input, line)) {
    std::istringstream is(line);
    // get the strings splitted by single space
    // brace-initialization with C++11
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    if (tokens.size() == 0) continue;
    if (tokens.size() == 2 && tokens[0] == "ngram") {
      std::string substring = tokens[1].substr(2);
      int32 count = std::stoi(substring); // get "123456" from "1=123456"
      counts_.push_back(count);
      order = std::stoi(tokens[1].substr(0));
      continue;
    }
    if (tokens.size() == 1 && tokens[0] == "\\1-grams:") {
      ngram_order_ = order; // ngram_order
      probs_.resize(ngram_order_);
      std::cout << "Ngram order is: " << ngram_order_ << std::endl;
    }
    // read current order
    if (tokens.size() == 1 && tokens[0] != "\\data\\" &&
        tokens[0] != "\\end\\") {
      order_current = std::stoi(tokens[0].substr(1,1));
      continue; // get the order info and skip processing this line
    }
    // read vocab and initialize probs of unigrams
    if (order_current == 1) {
      std::string word_s;
      if (tokens.back() != "</s>") {
        word_s = tokens.end()[-2];
        backoff_weight = std::stof(tokens.back());
      } else {
        word_s = tokens.back();
        backoff_weight = 0;
      }
      word = iter;
      vocab_.insert({word_s, word});
      iter++;
      vocab_size_++;
      if (iter == counts_[0]) {
        bool unigram_check = true;
        std::cout << "vocab size: " << vocab_size_ << std::endl;
      }
      HistType history;
      history.resize(0);
      log_prob = std::stof(tokens[0]);
      probs_pair = std::make_pair(log_prob, backoff_weight);
      probs_[order_current - 1][history].insert({word, probs_pair});
      continue;
    } 
    // read each ngram and its log-probs and back-off weights
    // read probs of order 1 to N - 1
    if (order_current < ngram_order_ && order_current > 1) {
      // case1: backoff_weights exist
      if ((tokens.size() > order_current + 1) && (tokens.back() != "</s>") && tokens[0] != "ngram") {
        // get the integer for word, the last second string in tokens
        std::string second_last = tokens.end()[-2];
        unordered_map<std::string, int32>::iterator it = vocab_.find(second_last);
        if (it != vocab_.end()) {
          word = it->second;
        } else {
          std::cout << "OOV word found: " << tokens.end()[-2] << std::endl;
        }
        int32 len_hist = tokens.size() - 3; // exclude the word, log-prob, and bow
        HistType history;
        for (int32 i = 1; i < len_hist + 1; i++) {
          unordered_map<std::string, int32>::iterator it = vocab_.find(tokens[i]);
          if (it != vocab_.end()) {
            history.push_back(it->second);
          } else {
            std::cout << "OOV found in history: " << tokens[i] << std::endl;
          }
        }
        assert (history.size() == order_current - 1);
        log_prob = std::stof(tokens[0]);
        backoff_weight = std::stof(tokens.back());
        probs_pair = std::make_pair(log_prob, backoff_weight);
        probs_[order_current - 1][history].insert({word, probs_pair});
        continue;
      }
      // case2: no backoff_weights
      if (tokens.size() == order_current + 1 && (tokens.back() == "</s>") && tokens[0] != "ngram") {
        unordered_map<std::string, int32>::iterator it = vocab_.find(tokens.back());
        if (it != vocab_.end()) {
          word = it->second;
        } 
        int32 len_hist = tokens.size() - 2; // exclude the word and log-prob
        HistType history;
        assert (len_hist > 0);
        for (int32 i = 1; i < len_hist + 1; i++) {
          unordered_map<std::string, int32>::iterator it = vocab_.find(tokens[i]);
          if (it != vocab_.end()) {
            history.push_back(it->second);
          } else {
            std::cout << "OOV found in history: " << tokens[i] << std::endl;
          }
        }
        assert (history.size() == order_current - 1);
        log_prob = std::stof(tokens[0]);
        backoff_weight = 0; // backoff_weight in log space should be 1 (no backoff)
        probs_pair = std::make_pair(log_prob, backoff_weight);
        probs_[order_current - 1][history].insert({word, probs_pair});
        continue;
      }
    } else if (order_current == ngram_order_) { // read probs of order N
      if (tokens.size() > 2) {
        std::string word_s = tokens.back();
        unordered_map<std::string, int32>::iterator it = vocab_.find(word_s);
        if (it != vocab_.end()) {
          word = it->second;
        }
        int32 len_hist = tokens.size() - 2; // exclude the word and log-prob
        HistType history;
        assert (len_hist > 0);
        for (int32 i = 1; i < len_hist + 1; i++) {
          unordered_map<std::string, int32>::iterator it = vocab_.find(tokens[i]);
          if (it != vocab_.end()) {
            history.push_back(it->second);
          } else {
            std::cout << "OOV found in history: " << tokens[i] << std::endl;
          }
        }
        log_prob = std::stof(tokens[0]);
        backoff_weight = 0; // backoff_weight in log space should be 1 (no backoff)
        probs_pair = std::make_pair(log_prob, backoff_weight);
        probs_[order_current - 1][history].insert({word, probs_pair});
        continue;
      }
    }
  }
  std::cout << "Finish reading ARPA-format file." << std::endl;
}

float NgramModel::GetProb(int32 order, const int32 word, const HistType& history) {
  float prob = 0.0;
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

float NgramModel::GetBackoffWeight(int32 order, const int32 word, const HistType& history) {
  float bow = 0.0;
  auto it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end()) {
    auto it2 = probs_[order - 1][history].find(word);
    if (it2 != probs_[order - 1][history].end()) {
      bow = (it2->second).second;
    }
  }
  return bow;
}

void NgramModel::ComputeWordPdf(const HistType& history, std::vector<float>* pdf) {
  int32 order = history.size();
  float prob = 0.0;
  for (int32 i = 0; i < vocab_size_; i++) {
    auto it = probs_[order].find(history);
    int32 word = i;
    if (it != probs_[order].end()) {
      auto it2 = probs_[order][history].find(word);
      if (it2 != probs_[order][history].end()) {
        prob = pow(10, (it2->second).first);
        (*pdf).push_back(prob);
      } else {
        HistType::const_iterator first = history.begin() + 1;
        HistType::const_iterator last = history.end();
        HistType h(first, last);
        int32 word_new = history.back();
        HistType::const_iterator last_new = history.end() - 1;
        HistType h_new(history.begin(), last_new);
        prob = pow(10, GetBackoffWeight(order, word_new, h_new)) *
               pow(10, GetProb(order, word, h));
        (*pdf).push_back(prob);
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
      (*pdf).push_back(prob);
    }
  }
}

// Get history weights
void NgramModel::ComputeHistoriesWeights() {
  for (auto it = histories_.begin(); it != histories_.end(); ++it) {
    HistType history(*(it));
    assert(history.size() <= ngram_order_);
    for (int32 i = 0; i < history.size() + 1; i++) {
      HistType h_tmp = history;
      float prob = 1.0 / histories_.size();
      while (h_tmp.size() > (history.size() - i)) {
        HistType::iterator last = h_tmp.end() - 1;
        HistType h(h_tmp.begin(), last);
        int32 word = h_tmp.back();
        prob *= pow(10, GetBackoffWeight(h_tmp.size(), word, h));
        h_tmp = h;
      }
      HistType::iterator begin = history.begin() + i;
      HistType h(begin, history.end());
      hists_weights_[h] += prob;
    }
  } 
  std::cout << "Size of hists_weights_ is: " << hists_weights_.size() << std::endl;
}

// Get weighted pdf
void NgramModel::ComputeWeightedPdf(std::vector<float>* pdf_w) {
  float prob = 0;
  (*pdf_w).resize(vocab_size_); // if do not do this, (*pdf_w)[word] += prob will get seg fault
  for (int32 i = 0; i < vocab_size_; i++) {
    for (auto it = hists_weights_.begin(); it != hists_weights_.end(); ++it) {
      HistType h(it->first);
      int32 order = h.size();
      auto it_hist = probs_[order].find(h);
      if (it_hist != probs_[order].end()) {
        int32 word = i;
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
            (*pdf_w)[word] += prob;
          } 
          else {
            prob = (it->second) * pow(10, probs_[order][h][word].first);
            (*pdf_w)[word] += prob;
          }
        }
      }
    } // end reading history
  } // end reading words
}

// sample a word that follows a pdf
int32 NgramModel::SampleWord(const std::vector<float>& pdf) {
  // generate a cdf from the given pdf 
  std::vector<std::pair<float, float> > cdf;
  float upper = 0;
  float lower = 0;
  std::pair<float, float> probs;
  for (int32 i = 0; i < pdf.size(); i++) {
    upper += pdf[i];
    lower = upper - pdf[i];
    probs = std::make_pair(lower, upper);
    cdf.push_back(probs);
  }
  float u = 1.0 * rand()/RAND_MAX;
  for (int32 i = 0; i < cdf.size(); i++) {
    if (cdf[i].first <= u < cdf[i].second) {
      return i;
    }
  }
}

// Sampling a word
void NgramModel::TestSampling(int32 iters) {
  ComputeHistoriesWeights();
  std::vector<float> pdf;
  ComputeWeightedPdf(&pdf);

  // Compute diff
  std::vector<float> pdf_est;
  pdf_est.resize(vocab_size_);
  int32 word;
  int32 count_nons = 0;
  for (int32 i = 0; i < iters; i++) {
    word = SampleWord(pdf);
    if (word > vocab_size_ || word < 0) {
      std::cout << "the next word is " << word << std::endl;
      count_nons += 1;
      continue;
    } else {
      pdf_est[word] += 1.0;
    }
  }
  // normalization
  float ed = 0;
  for (int32 i = 0; i < vocab_size_; i++) {
    pdf_est[word] /= iters;
    ed += pow(pdf_est[word] - pdf[word], 2); 
  }
  ed = pow(ed, 0.5);
  std::cout << "Run " << iters << " times, e distance (expect < 0.05) is " << ed << std::endl;
  std::cout << "Number of words OOV : " << count_nons << std::endl;
}

// Test the read-in language model
void NgramModel::TestReadingModel() {
  std::cout << "Testing model reading part..."<< std::endl;
  std::cout << "Vocab size is: " << vocab_size_ << std::endl;
  std::cout << "Ngram_order is: " << ngram_order_ << std::endl;
  assert(probs_.size() == counts_.size());
  for (int32 i = 0; i < ngram_order_; i++) {
    int32 size_ngrams = 0;
    std::cout << "Test: for order " << (i + 1) << std::endl;
    std::cout << "Expected number of " << (i + 1) << "-grams: " << counts_[i] << std::endl;
    for (auto it1 = probs_[i].begin(); it1 != probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (auto it2 = (probs_[i])[h].begin(); it2 != (probs_[i])[h].end(); ++it2) {
        size_ngrams++; // number of words given
      }
    }
    std::cout << "Read in number of " << (i + 1) << "-grams: " << size_ngrams << std::endl;
  }
  std::cout << "Assert sum of unigram probs equal to 1..." << std::endl;
  float prob_sum = 0.0;
  int32 count = 0;
  for (auto it1 = (probs_[0]).begin(); it1 != (probs_[0]).end();++it1) {
    HistType h(it1->first);
    for (auto it2 = (probs_[0])[h].begin(); it2 != (probs_[0])[h].end(); ++it2) {
      prob_sum += 1.0 * pow(10.0, (it2->second).first);
      count++;
    }
  }
  std::cout << "Number of total words: " << count << std::endl;
  std::cout << "Sum of unigram probs equal to " << prob_sum << std::endl;

  std::cout << "Assert sum of bigram probs given a history equal to 1..." << std::endl;
  prob_sum = 0.0;
  auto it1 = probs_[1].begin();
  HistType h(it1->first);
  for (auto it = vocab_.begin(); it != vocab_.end(); ++it) {
    auto it2 = probs_[1][h].find(it->second);
    if (it2 != probs_[1][h].end()) {
      prob_sum += 1.0 * pow(10, (it2->second).first);
    } else {
      prob_sum += pow(10, GetProb(2, it->second, h));
    }
  }
  std::cout << "Sum of bigram probs given a history equal to " << prob_sum << std::endl;

}

// Read histories of integers from a file
void NgramModel::ReadHistories(char* file) {
 std::ifstream data_input(file);
  if (!data_input.is_open()) {
    std::cerr << "error opening '" << file
              << "' for reading\n";
    exit(1);
  }
  std::string line;
  std::cout << "Start reading histories..." << std::endl;
  while (getline(data_input, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    HistType history;
    int32 word;
    for (int32 i = 0; i < tokens.size(); i++) {
      auto it = vocab_.find(tokens[i]);
      if (it != vocab_.end()) {
        word = it->second;
      } else {
        std::string word_s = "<unk>";
        auto it_unk = vocab_.find(word_s);
        assert (it_unk != vocab_.end());
        word = it_unk->second;
      }
      history.push_back(word);
    }
    if (history.size() >= ngram_order_) {
      // TODO: try slicing it later
      std::reverse(history.begin(), history.end());
      history.resize(ngram_order_ - 1);
      std::reverse(history.begin(), history.end());
    }
    histories_.push_back(history);
  }
  std::cout << "Finished reading histories." << std::endl;
}
