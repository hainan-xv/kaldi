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
#include "util/common-utils.h"

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

namespace kaldi {

typedef int32_t int32;

enum {
  kEps = 0,
  kDisambig,
  kBos, kEos, kUnk
};

typedef std::vector<int32> HistType;
typedef unordered_map<int32, std::pair<BaseFloat, BaseFloat> > WordToProbsMap; 
typedef unordered_map<HistType, WordToProbsMap, VectorHasher<int32> > NgramType;
typedef unordered_map<HistType, BaseFloat, VectorHasher<int32> > HistWeightsType;

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
  
  // This function returns the log probability of a ngram term from the ARPA LM
  // if it is found; it backoffs to the lower order model when the ngram term 
  // does not exist.
  BaseFloat GetProb(int32 order, int32 word, const HistType& history);

  // Get the back-off weight of a ngram in the read-in model
  BaseFloat GetBackoffWeight(int32 order, int32 word, const HistType& history);

  // Compute non-unigram output words and corresponding probs for given histories
  void ComputeOutputWords(std::vector<HistType> histories,
      unordered_map<int32, BaseFloat>* pdf_w);
  
  // Compute weighted pdf given all histories
  void ComputeWeightedPdf(HistWeightsType hists_weights, 
      std::vector<std::pair<int32, BaseFloat> >* weighted_pdf);
  
  // Get ngram order 
  int32 GetNgramOrder();

  void TestReadingModel();

  void TestProbs(std::istream &is, bool binary);

  void TestPdfsEqual();

  std::vector<HistType> ReadHistories(std::istream &is, bool binary);

 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable(); 
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete() {}

 private:
  // For test: randomly generate histories
  std::vector<HistType> RandomGenerateHistories();

  // Compute a pdf of words in the vocab given a history
  void ComputeWordPdf(const HistType& history, 
      std::vector<std::pair<int32, BaseFloat> >* pdf);
  
  // Compute weights of given histories
  HistWeightsType ComputeHistoriesWeights(std::vector<HistType> histories);

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
  
  // Test sentences 
  std::vector<std::vector<int32> > sentences_;
};

} // end of namespace kaldi
#endif
