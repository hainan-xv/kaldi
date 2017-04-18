// rnnlm/rnnlm-utils-test.cc

#include <math.h>
#include "rnnlm/rnnlm-utils.h"

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
    SampleWithoutReplacement(u, k, std::set<int>(), std::map<int, double>(), &samples);
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

      if (distance < 0.005) {
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
  UnitTestNChooseKSamplingConvergence(n, k, ones_size);
  // test when k = 1
  UnitTestNChooseKSamplingConvergence(n, 1, 0);
  // test when k = 2
  UnitTestNChooseKSamplingConvergence(n, 2, rand() % 1);
  // test when k = n
  ones_size = rand() % (n / 2);
  UnitTestNChooseKSamplingConvergence(n, n, ones_size);
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
  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<double> u(selection_probs);

  int N = iters;
  for (int i = 0; i < N; i++) {
    vector<std::pair<int, double> > samples;
    SampleWithoutReplacement(u, k, set<int>(), map<int, double>(), &samples);
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

  int N = iters;
  Timer t;
  t.Reset();
  double total_time;
  for (int i = 0; i < N; i++) {
    vector<std::pair<int, double> > samples;
    SampleWithoutReplacement(u, k, set<int>(), map<int, double>(), &samples);
  }
  total_time = t.Elapsed();
  KALDI_LOG << "Time test: Sampling " << k << " items from " << n <<
    " unigrams for " << N << " times takes " << total_time << " totally.";
}

}  // end namespace rnnlm
}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  using namespace rnnlm;
  int N = 1000;
  UnitTestSamplingConvergence();
  UnitTestSampleWithProbOne(N);
  UnitTestSamplingTime(N);
}

