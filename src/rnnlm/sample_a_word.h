// sample_a_word.h

// Copyright     2016  Ke Li

// See ../COPYING for clarification regarding multiple authors
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
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef SAMPLE_A_WORD_H_
#define SAMPLE_A_WORD_H_

#include <sys/time.h>
#include <unistd.h>

#ifdef _MSC_VER
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#elif __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#else
#include <tr1/unordered_map>
#include <tr1/unordered_set>
using std::tr1::unordered_map;
using std::tr1::unordered_set;
#endif

#include <cassert>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <queue>
#include <limits>

typedef int32_t int32;

/// A hashing function-object for vectors of ints.
struct IntVectorHasher {  // hashing function for vector<Int>.
  size_t operator()(const std::vector<int32> &x) const {
    size_t ans = 0;
    typename std::vector<int32>::const_iterator iter = x.begin(), end = x.end();
    for (; iter != end; ++iter) {
      ans *= kPrime;
      ans += *iter;
    }
    return ans;
  }
 private:
  static const int kPrime = 7853;
};

typedef std::vector<int32> HistType;
typedef unordered_map<int32, std::pair<float, float> > WordToProbsMap; 
typedef unordered_map<HistType, WordToProbsMap, IntVectorHasher> NgramType;
typedef unordered_map<HistType, float, IntVectorHasher> HistWeightsType;

class Timer {
 public:
  Timer() { Reset(); }

  void Reset() { gettimeofday(&this->time_start_, &time_zone_); }

  /// Returns time in seconds.
  double Elapsed() {
    struct timeval time_end;
    gettimeofday(&time_end, &time_zone_);
    double t1, t2;
    t1 =  static_cast<double>(time_start_.tv_sec) +
          static_cast<double>(time_start_.tv_usec)/(1000*1000);
    t2 =  static_cast<double>(time_end.tv_sec) +
          static_cast<double>(time_end.tv_usec)/(1000*1000);
    return t2-t1;
  }

 private:
  struct timeval time_start_;
  struct timezone time_zone_;
};

class NgramModel {
 public:
  // Constructor for testing
  NgramModel(char* arpa_file, char* histories_file);
  
  void TestReadingModel();

  void TestSampling(int32 iters);
  
 private:
  // This function returns the log probability of a ngram term from the ARPA LM
  // if it is found; it backoffs to the lower order model when the ngram term 
  // does not exist.
  float GetProb(int32 order, const int32 word, const HistType& history);

  // Get the back-off weight of a ngram in the read-in model
  float GetBackoffWeight(int32 order, const int32 word, const HistType& history);

  // Compute a pdf of words in the vocab given a history
  void ComputeWordPdf(const HistType& history, std::vector<float>* pdf);
  
  // Compute weights of given histories
  void ComputeHistoriesWeights();
  // Compute weighted pdf given all histories
  void ComputeWeightedPdf(std::vector<float>* weighted_pdf);
  
  // Sample the next word
  int32 SampleWord(const std::vector<float>& pdf);
  
  // Read the language model prob_ from stream
  // Called from constructor; Check the sum of unigrams
  void ReadARPAModel(char* arpa_file);
  
  void ReadHistories(char* file);

  // N-gram order of the read-in LM.
  int32 ngram_order_;
  
  // Counts of each ngram
  std::vector<int32> counts_; 

  // Vocab size
  int32 vocab_size_;

  // Vocab
  unordered_map<std::string, int32> vocab_;
   
  // N-gram probabilities.
  std::vector<NgramType> probs_;

  // Histories' weights
  HistWeightsType hists_weights_;
  
  // The given N Histories
  std::vector<HistType> histories_;
};

#endif
