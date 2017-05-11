#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

void DoSamplingInExamples(int num_samples, int ngram_order,
                          const vector<double>& unigram, const vector<double> &cdf,
                          NnetExample *egs) {
  int num_words_out = 0;
  int i = 0;
  KALDI_ASSERT (egs->io[0].name == "input");
  KALDI_ASSERT (egs->io[1].name == "output");

  vector<Index> &indexes = egs->io[i].indexes;
  int current_n = 0, current_t = 0;
  KALDI_ASSERT(current_n == indexes[0].n && current_t == indexes[0].t);

  int length = -1;
  for (int i = 1; i < indexes.size(); i++) {
    Index &index = indexes[i];
    if (index.n == current_n) {
      KALDI_ASSERT(index.t == ++current_t);
    } else {
      // just finished a sentence
      KALDI_ASSERT(index.t == 0);
      KALDI_ASSERT(index.n == ++current_n);
      if (length == -1) {
        length = current_t + 1;
      } else {
        KALDI_ASSERT(current_t + 1 == length);
      }
      current_t = 0;
    }
  }

  if (length == -1) {
    length = current_t + 1;
  }

  int32 minibatch_size = current_n + 1;

//  num_words_in = egs->io[0].features.NumCols();
  num_words_out = egs->io[1].features.NumCols();
  KALDI_ASSERT(unigram.size() == num_words_out);

  std::vector<std::vector<std::pair<int32, BaseFloat> > > samples;

  if (num_samples <= 0) {
    samples.resize(length, vector<std::pair<int32, BaseFloat> >());
    egs->io.push_back(nnet3::NnetIo("samples", 0, GeneralMatrix(SparseMatrix<BaseFloat>(unigram.size(), samples))));
    return;
  }

  vector<int32> input_words;
  vector<int32> output_words;
  SparseMatrixToVector(egs->io[0].features.GetSparseMatrix(), &input_words);
  SparseMatrixToVector(egs->io[1].features.GetSparseMatrix(), &output_words);

  samples.resize(length, vector<std::pair<int32, BaseFloat> >(minibatch_size));

  // v[t][n] is a vector representing the history of the input and n, t
  vector<vector<vector<int32> > > histories(length, vector<vector<int32> >(minibatch_size));
  vector<set<int32> > must_samples(length);

  for (int t = 0; t < length; t++) {
    for (int n = 0; n < minibatch_size; n++) {
      vector<int32> &history = histories[t][n];
      for (int j = 0; j < ngram_order && j <= t; j++) {
        history.push_back(input_words[length * n + t - j]);
      }
      must_samples[t].insert(output_words[length * n + t]);
    }
  }

  for (int t = 0; t < length; t++) {
    SampleWithoutReplacement(unigram, cdf, num_samples, must_samples[t], map<int, double>(), &samples[t]);
  }
//  if (num_samples != unigram.size()) {
  egs->io.push_back(nnet3::NnetIo("samples", 0, GeneralMatrix(SparseMatrix<BaseFloat>(unigram.size(), samples))));
//  }
}

void CheckValidGrouping(const vector<interval> &g, int k) {
  KALDI_ASSERT(g.size() >= k);
  // check sum
  double selection_sum = 0.0;
  double unigram_sum = 0.0;
  for (int i = 0; i < g.size(); i++) {
    selection_sum += std::min(1.0, g[i].selection_prob);
    unigram_sum += g[i].unigram_prob;
 
    KALDI_ASSERT(g[i].L < g[i].R);
    KALDI_ASSERT(g[i].selection_prob <= 1.0 || g[i].selection_prob > 5.0);
  }
 
  KALDI_ASSERT(ApproxEqual(selection_sum, k));
  KALDI_ASSERT(g[0].L == 0);
  for (int i = 1; i < g.size(); i++) {
    KALDI_ASSERT(g[i].L == g[i - 1].R);
  }
} 

void CheckValidGrouping(const vector<double> &u,
                        const vector<double> &cdf,
                        const std::set<int> &must_sample,
                        const std::map<int, double> &bigrams,
                        int k, const vector<interval> &g) {
  CheckValidGrouping(g, k); // check the self-consistency first

  double bigram_sum = 0.0;
  double unigram_sum = 0.0;

  unigram_sum = cdf[cdf.size() - 1];

  KALDI_ASSERT(ApproxEqual(1.0, unigram_sum));
  for (map<int, double>::const_iterator iter = bigrams.begin();
                                           iter != bigrams.end(); iter++) {
    bigram_sum += iter->second;
    unigram_sum -= u[iter->first];
  }

  double alpha = (1.0 - bigram_sum) / unigram_sum;

  double ratio = -1.0;
  for (int i = 0; i < g.size(); i++) {
    double ratio_ = -1;
    if (g[i].L + 1 == g[i].R) {
      // single world group
      int key = g[i].L;
      std::map<int, double>::const_iterator iter;
      if (must_sample.find(key) != must_sample.end()) {
        KALDI_ASSERT(g[i].selection_prob >= 1.0);
      } else if ((iter = bigrams.find(key)) != bigrams.end()) {
        if (g[i].selection_prob < 1.0) {
          ratio_ = iter->second / g[i].selection_prob;
        }
      } else {
        if (g[i].selection_prob < 1.0) {
          ratio_ = alpha * u[key] / g[i].selection_prob;
        }
      }
    } else {
      // it's a group of many words
      KALDI_ASSERT(g[i].selection_prob < 1.0);
      int start = g[i].L, end = g[i].R;
      ratio_ = alpha * (cdf[end] - cdf[start]) / g[i].selection_prob;
    }
    if (ratio < 0 && ratio_ > 0) {
      ratio = ratio_;
    } else if (ratio > 0 && ratio_ > 0) {
      KALDI_ASSERT(ApproxEqual(ratio_, ratio, 0.01));
    }
  }
}

// assume u is CDF of already sorted unigrams
void DoGroupingCDF(const vector<double> &u,
                   const vector<double> &cdf,
                   int k,
                   const set<int>& must_sample, const map<int, double> &bigrams,
                   vector<interval> *out) {

  KALDI_ASSERT(k >= must_sample.size());

  vector<double> bigram_probs;  // used in figuring out the max-allowed-unigram
  // it stores the bigram_probs of words that are not in the must_sample set
  // will be sorted later from greater to smaller

  KALDI_ASSERT(ApproxEqual(cdf[cdf.size() - 1], 1.0));

  double alpha = 0.0;
  double max_allowed_ngram_prob = 1.0 / (k - must_sample.size());

  // compute the value for alpha
  {
    double bigram_sum = 0.0;
    double unigram_sum = 1.0;

    for (map<int, double>::const_iterator iter = bigrams.begin();
                                          iter != bigrams.end();
                                          iter++) {
      bigram_sum += iter->second;
      unigram_sum -= u[iter->first];
      if (must_sample.find(iter->first) == must_sample.end()) {
        // only add to the vector if it is not in must-sample set
        bigram_probs.push_back(iter->second);
      }
    }

    alpha = (1.0 - bigram_sum) / unigram_sum;

    // now the n-gram probs for word i are bigram[i], or alpha * u[i]
    //////////////////////////////////////////////////////////////////////////

    double total_selection_wgt = k - must_sample.size();
    double total_ngram_wgt = 1.0;

    // delete all the weights that are in the must-sample set
    for (set<int>::const_iterator i = must_sample.begin(); i != must_sample.end(); i++) {
      map<int, double>::const_iterator ii = bigrams.find(*i);
      if (ii != bigrams.end()) {
        total_ngram_wgt -= ii->second;
      } else {
        total_ngram_wgt -= u[*i] * alpha;
      }
    }

    KALDI_ASSERT(total_ngram_wgt >= 0);
    
    sort(bigram_probs.begin(), bigram_probs.end(), std::greater<double>());

    int i = 0, j = 0;  // iterates 2 vectors like merge sort
    // i for bigram_probs, j for u
    // the following loop does the following: it iterators 2 vectors, each
    // time picking one with larger defacto probability and compute if we make
    // the cutoff there, will the largest selection-prob be > 1.
    // it keeps doing so until the right cut-off point is found
    while (true) {
      // find the first unigram that counts (neither in must-sample nor in bigram)
      while (must_sample.find(j) != must_sample.end() || bigrams.find(j) != bigrams.end()) {
        j++;
      }
      // no need to do this for bigram since it only stores one's that are not
      // in must-sample

      if (j == u.size()) {
        max_allowed_ngram_prob = 1.0 / u.size();
        break;
      }

      // now neither i or j has "special" probs; both only depend on the ngram probs

      double p;
      // p takes the value of the max of bigram and unigram
      if (i < bigram_probs.size() && bigram_probs[i] > alpha * u[j]) {
        p = bigram_probs[i];
        i++;
      } else {
        p = alpha * u[j];
        j++;
      }
      if (p / total_ngram_wgt * total_selection_wgt > 1.0) {
        // needs to look further
        total_ngram_wgt -= p;
        total_selection_wgt -= 1.0;
        if (total_ngram_wgt < 0) {
//          KALDI_LOG << "the number should be close to 0.0" << total_ngram_wgt;
          KALDI_ASSERT(-total_ngram_wgt < 0.0000001);
          total_ngram_wgt = 0.0;
        }
      } else {
        max_allowed_ngram_prob = total_ngram_wgt / total_selection_wgt;
        KALDI_ASSERT(max_allowed_ngram_prob > 0);
        break;
      }
    }
  }

  set<int>::const_iterator must_sample_iter = must_sample.begin();
  map<int, double>::const_iterator bigram_iter = bigrams.begin();

  for (int i = 0; i < u.size();) {
    // a must-sample word
    if (must_sample_iter != must_sample.end() && *must_sample_iter == i) {
      out->push_back(interval(i, i + 1, -1.0, ONE));
      if (bigram_iter != bigrams.end() && *must_sample_iter == bigram_iter->first) {
        bigram_iter++;
      }
      must_sample_iter++;
      i++;
    } else if (bigram_iter != bigrams.end() && bigram_iter->first == i) {
      double p = bigram_iter->second / max_allowed_ngram_prob;
      if (p > 1.0) {
        p = ONE;
      }
      out->push_back(interval(i, i + 1, bigram_iter->second, p));
      bigram_iter++;
      i++;
    } else {
      double u_i = u[i] * alpha;

      int n = max_allowed_ngram_prob / u_i; // since we know the original unigram is sorted we could at least have n
      int group_end = i + n;

      int index_upper_bound = INT_MAX;
      if (must_sample_iter != must_sample.end()) {
        index_upper_bound = std::min(index_upper_bound, *must_sample_iter);
      }
      if (bigram_iter != bigrams.end()) {
        index_upper_bound = std::min(index_upper_bound, bigram_iter->first);
      }

      group_end = std::min(group_end, int(u.size()));
      group_end = std::min(group_end, index_upper_bound);

      KALDI_ASSERT(group_end >= i);

      if (group_end == i) {
        out->push_back(interval(i, i + 1, -1, ONE));
        i++;
      } else {
        double uni_prob = cdf[group_end];
        if (i != 0) {
          uni_prob -= cdf[i];
        }
        uni_prob *= alpha;
        double selection_prob = uni_prob / max_allowed_ngram_prob;

        if (selection_prob > 1.0) {
          KALDI_ASSERT(ApproxEqual(selection_prob, 1.0));
          selection_prob = 1.0;
        }
        KALDI_ASSERT(selection_prob > 0 && selection_prob <= 1.0);
        out->push_back(interval(i, group_end, uni_prob, selection_prob));
        i = group_end;
      }
    }
  }
}

// return an int i in [L, R - 1], w/ probs porportional to their pdf's
int SelectOne(const vector<double> &cdf, int L, int R) {
  double p = RandUniform() * (cdf[R] - cdf[L]) + cdf[L];

  int index = -1;

  // it actually happened in the test that they're equal
  if (p <= cdf[L]) {
    index = L;
  }
  else if (p >= cdf[R]) {
    index = R - 1;
  } else {
    int i1 = L;           // >= i1
    int i2 = R;  // <  i2
    int mid = -1;
    while (true) {
      KALDI_ASSERT(i1 < i2);
      mid = (i1 + i2) / 2;
      if (cdf[mid + 1] >= p && cdf[mid] < p) {
        index = mid;
        break;
      } else if (cdf[mid] >= p) {
        i2 = mid;
      } else {
        i1 = mid + 1;
      }
    }
  }
  return index;
  
}

void VectorToSparseMatrix(const vector<int32> &v,
                          int dim,
                          SparseMatrix<BaseFloat> *sp) {
  std::vector<std::vector<std::pair<MatrixIndexT, BaseFloat> > > pairs;
  for (int i = 0; i < v.size(); i++) {
    std::vector<std::pair<MatrixIndexT, BaseFloat> > this_row;
    this_row.push_back(std::make_pair(v[i], 1.0));
    pairs.push_back(this_row);
  }
  *sp = SparseMatrix<BaseFloat>(dim, pairs);
}

void SparseMatrixToVector(const SparseMatrix<BaseFloat> &sp,
                          vector<int32> *v) {
  int k = sp.NumRows();
  v->resize(k);
  for (int i = 0; i < k; i++) {
    const SparseVector<BaseFloat> &sv = sp.Row(i);                              
    int non_zero_index = -1;                                                    
    KALDI_ASSERT(sv.NumElements() == 1);
    sv.Max(&non_zero_index); 
    (*v)[i] = non_zero_index;
  }
}

bool LargerThan(const std::pair<int, BaseFloat> &t1,
                const std::pair<int, BaseFloat> &t2) {
  return t1.second > t2.second;
}

vector<string> SplitByWhiteSpace(const string &line) {
  std::stringstream ss(line);
  vector<string> ans;
  string word;
  while (ss >> word) {
    ans.push_back(word);
  }
  return ans;
}

void ReadWordlist(string filename, unordered_map<string, int> *out) {
  unordered_map<string, int>& ans = *out;
  ifstream ifile(filename.c_str());
  string word;
  int id;

  while (ifile >> word >> id) {
    ans[word] = id;
  }
}

void ReadUnigram(string f, vector<double> *u) {
  ifstream ifile(f.c_str());
  int id;
  BaseFloat prob;
  BaseFloat sum = 0.0;
  while (ifile >> id >> prob) {
    KALDI_ASSERT(id == u->size());
    (*u).push_back(prob);
    sum += prob;
  }

  for (int i = 0; i < u->size(); i++) {
    (*u)[i] /= sum;
  }
}

void GetEgsFromSent(const vector<int>& word_ids_in, int input_dim,
                    const vector<int>& word_ids_out, int output_dim, NnetExample *out) {
  SparseMatrix<BaseFloat> input_frames(word_ids_in.size(), input_dim);

  for (int j = 0; j < word_ids_in.size(); j++) {
    vector<std::pair<MatrixIndexT, BaseFloat> > pairs;
    pairs.push_back(std::make_pair(word_ids_in[j], 1.0));
    SparseVector<BaseFloat> v(input_dim, pairs);
    input_frames.SetRow(j, v);
  }

  NnetExample &eg = *out;
  eg.io.push_back(nnet3::NnetIo("input", 0, GeneralMatrix(input_frames)));

  Posterior posterior;
  for (int i = 0; i < word_ids_out.size(); i++) {
    vector<std::pair<int32, BaseFloat> > p;
    p.push_back(std::make_pair(word_ids_out[i], 1.0));
    posterior.push_back(p);
  }

  eg.io.push_back(nnet3::NnetIo("output", output_dim, 0, posterior));
}

void SampleWithoutReplacement(const vector<double> &u, const vector<double>& cdf, int n,
                              const set<int>& must_sample, const map<int, double> &bigrams,
                              vector<std::pair<int, BaseFloat> > *out) {
  // out.first is the word and out.second is P(choose that word)
  KALDI_ASSERT(u.size() + 1 == cdf.size());
  if (n == u.size()) {
    out->resize(n);
    for (int i = 0; i < n; i++) {
      (*out)[i].first = i;
      (*out)[i].second = 1.0;
    }
    return;
  }

// assume u is sorted from large to small
  vector<interval> g;
  DoGroupingCDF(u, cdf, n, must_sample, bigrams, &g);

  vector<std::pair<int, double> > group_u(g.size());
  for (int i = 0; i < g.size(); i++) {
    group_u[i].first = i;
    group_u[i].second = g[i].selection_prob;
  }
  SampleWithoutReplacement_(group_u, n, out);

//  std::cout << "selected groups are: ";
//  for (int i = 0; i < out->size(); i++) {
//    std::cout << (*out)[i] << " ";
//  }
//  std::cout << std::endl;

  for (int i = 0; i < out->size(); i++) {
    if (g[(*out)[i].first].L + 1 < g[(*out)[i].first].R) { // is a group of many
      int index = SelectOne(cdf, g[(*out)[i].first].L, g[(*out)[i].first].R);
//      KALDI_ASSERT((*out)[i].second * u[index] / g[(*out)[i].first].unigram_prob == u[index] * n);
      (*out)[i].second = (*out)[i].second * u[index] / g[(*out)[i].first].unigram_prob;
      (*out)[i].first = index;

    } else {
//      KALDI_ASSERT((*out)[i].second == g[(*out)[i].first].selection_prob);
      if ((*out)[i].second > 1.0) {
        KALDI_ASSERT((*out)[i].second == ONE);
        (*out)[i].second = 1.0;
      }
      (*out)[i].first = g[(*out)[i].first].L;
//      map<int, double>::const_iterator iter = bigrams.find((*out)[i].first);
//      if (iter != bigrams.end()) {
//        (*out)[i].second = iter->second * n;
//      } else if (must_sample.find((*out)[i].first) != must_sample.end()) {
//        (*out)[i].second = 1.0;
//      }
    }
//    KALDI_ASSERT((*out)[i].second <= 1.0);
  }
//  cout << "selected words are: ";
//  for (int i = 0; i < out->size(); i++) {
//    cout << (*out)[i] << " ";
//  }
//  cout << endl;
}

// u is vector of <word, prob> pairs
// select words according to these probs into out
void SampleWithoutReplacement_(vector<std::pair<int, double> > u, int n,
                               vector<std::pair<int, BaseFloat> > *out) {
  sort(u.begin(), u.end(), LargerThan);

  KALDI_ASSERT(n != 0);

  vector<std::pair<int, BaseFloat> >& ans(*out);
  ans.resize(n);

  double tot_weight = 0;

  for (int i = 0; i < n; i++) {
    tot_weight += std::min(1.0, u[i].second);
    ans[i].first = i;
  }

  for (int k = n; k < u.size(); k++) {
    tot_weight += std::min(1.0, u[k].second);
    double pi_k1_k1 = u[k].second / tot_weight * n;

    if (u[k].second > 5.0) {
      pi_k1_k1 = 1.0;
    }
    if (pi_k1_k1 > 1) {
      KALDI_ASSERT(false); // never gonna happen in our setup since sorted
      pi_k1_k1 = 1;  // must add
    } else {
      BaseFloat p = RandUniform();
      if (p > pi_k1_k1) {
        continue;
      }
    }

    vector<double> R(n);
    // fill up R
    {
      double Lk = 0;
      double Tk = 0;
      for (int i = 0; i < n; i++) {
        double pi_k_i = u[ans[i].first].second /
                   (tot_weight - std::min(1.0, u[k].second)) * n;
        double pi_k1_i = u[ans[i].first].second / tot_weight * n;

        if (u[ans[i].first].second >= 5.0) {
          pi_k_i = pi_k1_i = 1;
        }

        if (pi_k_i >= 1 && pi_k1_i >= 1) {
          // case A
          R[i] = 0;
          Lk++;
        } else if (pi_k_i >= 1 && pi_k1_i < 1) {
          // case B
          R[i] = (1 - pi_k1_i) / pi_k1_k1;
          Tk += R[i];
          Lk++;
        } else if (pi_k_i < 1 && pi_k1_i < 1) { // case C we will handle in another loop
        } else {
          KALDI_ASSERT(false);
        }
      }

      double sum = 0;
      for (int i = 0; i < n; i++) {
        double pi_k_i = u[ans[i].first].second /
            (tot_weight - std::min(1.0, u[k].second)) * n;
        double pi_k1_i = u[ans[i].first].second / tot_weight * n;

        if (pi_k_i < 1 && pi_k1_i < 1) {
          // case C
          R[i] = (1 - Tk) / (n - Lk);
        }
        sum += R[i];
      }
      KALDI_ASSERT(ApproxEqual(sum, 1.0));
    }

    vector<double> cdf(R);
    double *cdf_ptr = &(cdf[0]);
    double cdf_sum = 0.0;
    for (int32 i = 0, size = cdf.size(); i < size; i++) {
      cdf_sum += cdf_ptr[i];
      cdf_ptr[i] = static_cast<double>(cdf_sum);
    }

    double p = RandUniform() * cdf[cdf.size() - 1];
    // binary search

    int index = -1;
    {
      int i1 = 0;           // >= i1
      int i2 = cdf.size();  // <  i2
      int mid = -1;
      while (true) {
        KALDI_ASSERT(i1 < i2);
        mid = (i1 + i2) / 2;
        if (cdf[mid] >= p && cdf[mid - 1] < p) {
          // found it
          // no need to worry mid - 1 < 0 since there are at least one word
          // with prob = 1 -> R[0] == 0.0 and mid would never be 0
          index = mid;
          break;
        } else if (cdf[mid] > p) {
          i2 = mid;
        } else {
          i1 = mid + 1;
        }
      }
    }

    KALDI_ASSERT(R[index] != 0);
    KALDI_ASSERT(u[index].second < 1.0);
    ans[index].first = k;

  }

  //  change to the correct indexes
  for (int i = 0; i < ans.size(); i++) {
    // need to change the second first
    ans[i].second = u[ans[i].first].second;
    ans[i].first = u[ans[i].first].first;
  }
}

 void NormalizeVec(int k, const set<int>& ones, vector<double> *probs) {
   KALDI_ASSERT(ones.size() < k);
   // first check the unigrams add up to 1
   BaseFloat sum = 0;
   for (int i = 0; i < probs->size(); i++) {
     sum += (*probs)[i];
   }
   KALDI_ASSERT(ApproxEqual(sum, 1.0));
   
 //  set the 1s to be 10.0 to avoid numerical issues
   for (set<int>::const_iterator iter = ones.begin(); iter != ones.end(); iter++) {
     sum -= (*probs)[*iter];
     (*probs)[*iter] = 10.0;  // mark the ones
   }
 
   // distribute the remaining probs
   BaseFloat maxx = 0.0;
   for (int i = 0; i < probs->size(); i++) {
     BaseFloat t = (*probs)[i];
     if (t < 5.0) {
       maxx = std::max(maxx, t);
     }
   }
 
 //  KALDI_LOG << "sum should be smaller than 1.0, larger than 0.0: " << sum;
 //  KALDI_LOG << "max is " << maxx;
 //  KALDI_ASSERT(false);
 
   if (maxx / sum * (k - ones.size()) <= 1.0) {
 //    KALDI_LOG << "no need to interpolate";
     for (int i = 0; i < probs->size(); i++) {
       double &t = (*probs)[i];
       if (t > 5.0) {
         continue;
       }
       t = t * (k - ones.size()) * 1.0 / sum;
     }
   } else {
 //    KALDI_LOG << "need to interpolate";
     // now we want to interpolate with a uniform prob of
     // 1.0 / (probs->size() - ones.size()) * (k - ones.size()) s.t. max is 1
     // (total prob = k - one.size() and we have probs->size() - ones.size() words)
     // w a + (1 - w) b = 1
     // ===> w = (1 - b) / (a - b)
     BaseFloat a = maxx / sum * (k - ones.size());
 //    BaseFloat b = (k - ones.size()) / (probs->size() - ones.size());
     BaseFloat b = 1.0 / (probs->size() - ones.size()) * (k - ones.size());
     BaseFloat w = (1.0 - b) / (a - b);
     KALDI_ASSERT(w >= 0.0 && w <= 1.0);
 
     for (int i = 0; i < probs->size(); i++) {
       double &t = (*probs)[i];
       if (t > 5.0) {
         continue;
       }
       t = w * t / sum * (k - ones.size()) + (1.0 - w) * b;
     }
   }
 
   for (set<int>::const_iterator iter = ones.begin(); iter != ones.end(); iter++) {
     KALDI_ASSERT((*probs)[*iter] > 5.0);
 //    (*probs)[*iter] = 2.0;  // to avoid numerical issues
   }
   
   sum = 0.0;
   for (int i = 0; i < probs->size(); i++) {
     BaseFloat t = (*probs)[i];
     sum += std::min(BaseFloat(1.0), t);
 //    KALDI_LOG << "sum is " << sum;
   }
   KALDI_ASSERT(ApproxEqual(sum, k));
 
 }

void ComponentDotProducts(const LmNnet &nnet1,
                          const LmNnet &nnet2,
                          VectorBase<BaseFloat> *dot_prod) {

  Vector<BaseFloat> v1(dot_prod->Dim() - 2);
  nnet3::ComponentDotProducts(nnet1.Nnet(), nnet2.Nnet(), &v1);
  dot_prod->Range(0, dot_prod->Dim() - 2).CopyFromVec(v1);

  int32 dim = dot_prod->Dim();
  dot_prod->Data()[dim - 2] = nnet1.InputLayer()->DotProduct(*nnet2.InputLayer());
  dot_prod->Data()[dim - 1] = nnet1.OutputLayer()->DotProduct(*nnet2.OutputLayer());
}

std::string PrintVectorPerUpdatableComponent(const LmNnet &lm_nnet,
                                             const VectorBase<BaseFloat> &vec) {
  const nnet3::Nnet &nnet = lm_nnet.Nnet();
  std::ostringstream os;
  os << "[ ";
  KALDI_ASSERT(NumUpdatableComponents(nnet) == vec.Dim() - 2);
  int32 updatable_c = 0;
  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const nnet3::Component *comp = nnet.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      const std::string &component_name = nnet.GetComponentName(c);
      os << component_name << ':' << vec(updatable_c) << ' ';
      updatable_c++;
    }
  }
  os << "rnnlm input: "  << vec(updatable_c) << " ";
  os << "rnnlm output: " << vec(updatable_c + 1) << " ";
  KALDI_ASSERT(updatable_c + 2 == vec.Dim());
  os << ']';
  return os.str();
}

} // namespace rnnlm
} // namespace kaldi
