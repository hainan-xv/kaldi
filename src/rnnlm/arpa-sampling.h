// arpa_sampling.h

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

#ifndef ARPA_SAMPLING_H_
#define ARPA_SAMPLING_H_

#include <sys/time.h>
#include <unistd.h>
#include "lm/arpa-file-parser.h"
#include "fst/fstlib.h"

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

namespace kaldi {

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

// Predefine some symbol values, because any integer is as good than any other.
enum {
  kEps = 0,
  kDisambig,
  kBos, kEos, kUnk
};

typedef std::vector<int32> HistType;
typedef unordered_map<int32, std::pair<BaseFloat, BaseFloat> > WordToProbsMap; 
typedef unordered_map<HistType, WordToProbsMap, IntVectorHasher> NgramType;
typedef unordered_map<HistType, BaseFloat, IntVectorHasher> HistWeightsType;

class ArpaSampling : public ArpaFileParser {
 public:
  // constructor
  explicit ArpaSampling(ArpaParseOptions options, fst::SymbolTable* symbols)
     : ArpaFileParser(options, symbols) { 
       ngram_order_ = 0;
       num_words_ = 0;
       bos_symbol_ = "<s>";
       eos_symbol_ = "</s>";
       unk_symbol_ = "<unk>";
  }
  // Compute the probability of a given sentence with ngram_order LM
  BaseFloat ComputeSentenceProb(const std::vector<int32>& test_sentence);
  
  // Test the read-in model by computing probs of all sentences with ngram_order LM
  BaseFloat ComputeAllSentencesProb(const std::vector<std::vector<int32> >& test_sentences);
  
  void TestReadingModel();

  void TestProbs(std::istream &is, bool binary);

  void TestSampling();
  
  void TestPdfsEqual();

  // print history
  void PrintHist(const HistType& h);
  
  void ReadHistories(std::istream &is, bool binary);

  void ReadSentences(std::istream &is, std::vector<std::vector<int32> >* sentences);
  
 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable(); 
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete() {}

 private:
  // This function returns the log probability of a ngram term from the ARPA LM
  // if it is found; it backoffs to the lower order model when the ngram term 
  // does not exist.
  BaseFloat GetProb(int32 order, int32 word, const HistType& history);

  // Get the back-off weight of a ngram in the read-in model
  BaseFloat GetBackoffWeight(int32 order, int32 word, const HistType& history);

  // Compute a pdf of words in the vocab given a history
  void ComputeWordPdf(const HistType& history, std::vector<std::pair<int32, BaseFloat> >* pdf);
  
  // Compute weights of given histories
  void ComputeHistoriesWeights();

  // Compute weighted pdf given all histories
  void ComputeWeightedPdf(std::vector<std::pair<int32, BaseFloat> >* weighted_pdf);
  
  // Sample the next word
  int32 SampleWord(const std::vector<std::pair<int32, BaseFloat> >& pdf);

  // N-gram order of the read-in LM.
  int32 ngram_order_;
  
  // num_words
  int32 num_words_;

  // Bos symbol
  std::string bos_symbol_;

  // Eos symbol
  std::string eos_symbol_;

  // Unk symbol
  std::string unk_symbol_;

  // Vocab
  std::vector<std::pair<std::string, int32> > vocab_;

  // Counts of each ngram
  std::vector<int32> ngram_counts_;

  // N-gram probabilities.
  std::vector<NgramType> probs_;

  // Histories' weights
  HistWeightsType hists_weights_;
  
  // The given N Histories
  std::vector<HistType> histories_;
  
  // Test sentences 
  std::vector<std::vector<int32> > sentences_;
};

} // end of namespace kaldi
#endif
