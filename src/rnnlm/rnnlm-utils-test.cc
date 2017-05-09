// rnnlm/rnnlm-utils-test.cc

#include "rnnlm/rnnlm-utils.h"
#include "arpa-sampling.h"

#include <math.h>
#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

namespace kaldi {
namespace rnnlm {

void PrepareVector(int n, int ones_size, std::set<int>* must_sample_set,
                   vector<double>* selection_probs) {
  double prob = 0;
  double prob_sum = 0;
  for (int i = 0; i < n; i++) {
    prob = RandUniform();
    prob_sum += prob;
    (*selection_probs).push_back(prob);
  }
  for (int i = 0; i < n; i++) {
    (*selection_probs)[i] /= prob_sum;
  }

  sort(selection_probs->begin(), selection_probs->end(), std::greater<double>());

  for (int i = 0; i < ones_size; i++) {
    (*must_sample_set).insert(rand() % n);
  }
}

void UnitTestNChooseKSamplingConvergence(int n, int k, int ones_size) {
  std::set<int> must_sample_set;
  vector<double> selection_probs;
  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);
//  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<double> &u(selection_probs);

  vector<double> cdf(u.size() + 1);
  cdf[0] = 0;
  for (int i = 1; i < cdf.size(); i++) {
    cdf[i] = cdf[i - 1] + u[i - 1];
  }

  // normalize the selection_probs
  double sum = 0;
  for (int i = 0; i < u.size(); i++) {
    sum += std::min(double(1.0), selection_probs[i]);
  }
  KALDI_ASSERT(ApproxEqual(sum, 1.0));

  vector<double> samples_counts(u.size(), 0);
  int count = 0;
  for (int i = 0; ; i++) {
//    KALDI_LOG << "count is " << count;
    count++;
    vector<std::pair<int, double> > samples;
    SampleWithoutReplacement(u, cdf, k, std::set<int>(), std::map<int, double>(), &samples);
    for (int j = 0; j < samples.size(); j++) {
      samples_counts[samples[j].first] += 1;
    }
    // update Euclidean distance between the two pdfs every 1000 iters
    if (count % 100 == 0) {
      double distance = 0;
      vector<double> samples_probs(u.size());
      for (int j = 0; j < samples_probs.size(); j++) {
        samples_probs[j] = samples_counts[j] / (count * k);
      }
      for (int j = 0; j < u.size(); j++) {
        distance += pow(samples_probs[j] - selection_probs[j], 2);
      }
      distance = sqrt(distance);

      KALDI_LOG << "distance after " << count << " runs is " << distance;

      if (distance < 0.01) {
        KALDI_LOG << "Sampling convergence test: passed for sampling " << k <<
          " items from " << n << " unigrams";
        break;
      }
    }
    // if the Euclidean distance is small enough, break the loop
  }
}

void UnitTestSamplingConvergence() {
  // number of unigrams
  int n = rand() % 10000 + 100;
  // sample size
  int k;
  // number of ones
  int ones_size;
  ones_size = rand() % (n / 2);
  k = rand() % (n - ones_size) + ones_size + 1;
//  KALDI_LOG << "testing choose k";
//  UnitTestNChooseKSamplingConvergence(n, k, ones_size);
  KALDI_LOG << "testing choose 1";
  UnitTestNChooseKSamplingConvergence(n, 1, 0);
  KALDI_LOG << "testing choose 2";
  UnitTestNChooseKSamplingConvergence(n, 2, 0);
  // test when k = n
  KALDI_LOG << "testing choose all";
  UnitTestNChooseKSamplingConvergence(n, n, 0);
}

// test that probabilities 1.0 are always sampled
void UnitTestSampleWithProbOne(int iters) {
  // number of unigrams
  int n = rand() % 1000 + 100;
  // generate a must_sample_set with ones
  int ones_size = rand() % (n / 2);
  std::set<int> must_sample_set;
  vector<double> selection_probs;

  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);

//  KALDI_LOG << "Must sample: ";
//  for (std::set<int>::iterator iter = must_sample_set.begin();
//                               iter != must_sample_set.end();
//                               iter++) {
//    KALDI_LOG << *iter << " ";
//  }

  // generate a random number k from ones_size + 1 to n
  int k = rand() % (n - ones_size) + ones_size + 1;
//  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<double> u(selection_probs);
  vector<double> cdf(u.size() + 1);
  cdf[0] = 0;
  for (int i = 1; i < cdf.size(); i++) {
    cdf[i] = cdf[i - 1] + u[i - 1];
  }

  int N = iters;
  for (int i = 0; i < N; i++) {
    vector<std::pair<int, double> > samples;
    SampleWithoutReplacement(u, cdf, k, must_sample_set, map<int, double>(), &samples);
    if (must_sample_set.size() > 0) {
      // assert every item in must_sample_set is sampled
      for (set<int>::iterator it = must_sample_set.begin(); it != must_sample_set.end(); ++it) {
        KALDI_ASSERT(std::find(samples.begin(), samples.end(), std::make_pair(*it, double(1.0))) != samples.end());
      }
    }
  }
  KALDI_LOG << "Test sampling with prob = 1.0 successful";
}

void UnitTestSamplingTime(int iters) {
  // number of unigrams
  int n = rand() % 1000 + 100;
  // generate a must_sample_set with ones
  int ones_size = rand() % (n / 2);
  std::set<int> must_sample_set;
  vector<double> selection_probs;

  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);

  // generate a random number k from ones_size + 1 to n
  int k = rand() % (n - ones_size) + ones_size + 1;
//  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<double> &u(selection_probs);
  vector<double> cdf(u.size() + 1);
  cdf[0] = 0;
  for (int i = 1; i < cdf.size(); i++) {
    cdf[i] = cdf[i - 1] + u[i - 1];
  }

  int N = iters;
  Timer t;
  t.Reset();
  double total_time;
  for (int i = 0; i < N; i++) {
    vector<std::pair<int, double> > samples;
    SampleWithoutReplacement(u, cdf, k, set<int>(), map<int, double>(), &samples);
  }
  total_time = t.Elapsed();
  KALDI_LOG << "Time test: Sampling " << k << " items from " << n <<
    " unigrams for " << N << " times takes " << total_time << " totally.";
}

}  // end namespace rnnlm
}  // end namespace kaldi.

int main(int argc, char **argv) {
  using namespace kaldi;
  using namespace rnnlm;
  int N = 10000;
  UnitTestSamplingConvergence();
  UnitTestSampleWithProbOne(N);
  UnitTestSamplingTime(N);

  const char *usage = "";
  ParseOptions po(usage);
  po.Read(argc, argv);
  std::string arpa_file = po.GetArg(1), history_file = po.GetArg(2);
  
  ArpaParseOptions options;
  fst::SymbolTable symbols;
  // Use spaces on special symbols, so we rather fail than read them by mistake.
  symbols.AddSymbol(" <eps>", kEps);
  // symbols.AddSymbol(" #0", kDisambig);
  options.bos_symbol = symbols.AddSymbol("<s>", kBos);
  options.eos_symbol = symbols.AddSymbol("</s>", kEos);
  options.unk_symbol = symbols.AddSymbol("<unk>", kUnk);
  options.oov_handling = ArpaParseOptions::kAddToSymbols;
  ArpaSampling mdl(options, &symbols);
  
  bool binary;
  Input k1(arpa_file, &binary);
  mdl.Read(k1.Stream(), binary);
  mdl.TestReadingModel();
   
  Input k2(history_file, &binary);
  std::vector<HistType> histories;
  histories = mdl.ReadHistories(k2.Stream(), binary);
  unordered_map<int32, BaseFloat> pdf_hist_weight;
  mdl.ComputeOutputWords(histories, &pdf_hist_weight);
  // command for running the test binary: ./test-binary arpa-file history-file
  // arpa-file is the ARPA-format language model
  // history-file has lines of histories, one history per line

  // this test can be slow
  /*
  KALDI_LOG << "Start weighted histories test...";
  for (int i = 0; i < N / 100; i++) {
    mdl.TestPdfsEqual(); 
  }
  KALDI_LOG << "Successfuly pass the test.";
  */
  return 0;
}
