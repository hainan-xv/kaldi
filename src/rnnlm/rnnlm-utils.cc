#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {


void CheckValidGrouping(const vector<interval> &g, int k) {
  KALDI_ASSERT(g.size() >= k);
  // check sum
  BaseFloat selection_sum = 0.0;
  BaseFloat unigram_sum = 0.0;
  for (int i = 0; i < g.size(); i++) {
    selection_sum += std::min(BaseFloat(1.0), g[i].selection_prob);
    unigram_sum += g[i].unigram_prob;

    KALDI_ASSERT(g[i].L < g[i].R);
    KALDI_ASSERT(g[i].selection_prob <= 1.0 || g[i].selection_prob > 5.0);
  }
  KALDI_ASSERT(ApproxEqual(selection_sum, k));
//  KALDI_ASSERT(ApproxEqual(unigram_sum, 1.0));
  KALDI_ASSERT(g[0].L == 0);
//  KALDI_ASSERT(g[g.size() - 1].R == 0);
  for (int i = 1; i < g.size(); i++) {
    KALDI_ASSERT(g[i].L == g[i - 1].R);
  }
} 

// assume u is CDF of already sorted unigrams
void DoGroupingCDF(const vector<std::pair<int, BaseFloat> > &u,
                   const vector<BaseFloat> &cdf,
                   int k,
                   const set<int>& must_sample, const map<int, BaseFloat> &bigrams,
                   vector<interval> *out) {

  KALDI_ASSERT(k >= must_sample.size());

  vector<double> bigram_probs;  // used in figuring out the max-allowed-unigram

  KALDI_ASSERT(ApproxEqual(cdf[cdf.size() - 1], 1.0));

  BaseFloat alpha = 0.0;
  BaseFloat max_allowed_ngram_prob = 1.0 / (k - must_sample.size());

  // compute the value for alpha
  {
    BaseFloat bigram_sum = 0.0;
    BaseFloat unigram_sum = 1.0;

    for (map<int, BaseFloat>::const_iterator iter = bigrams.begin();
                                             iter != bigrams.end();
                                             iter++) {
      bigram_sum += iter->second;
      unigram_sum -= u[iter->first].second;
      if (must_sample.find(iter->first) == must_sample.end()) {
        bigram_probs.push_back(iter->second);
      }
    }

    alpha = (1.0 - bigram_sum) / unigram_sum;
//    KALDI_LOG << "bigram_sum is " << bigram_sum;
//    KALDI_LOG << "unigram_sum is " << unigram_sum;
//    KALDI_LOG << "alpha is " << alpha;
    // now the n-gram probs for word i are bigram[i], or alpha * u[i] (standard arpa file rules)
    //////////////////////////////////////////////////////////////////////////

    BaseFloat total_selection_wgt = k - must_sample.size();
    BaseFloat total_ngram_wgt = 1.0;

//    KALDI_LOG << "total selection weight is: " << total_selection_wgt;
//    KALDI_LOG << "total unigram weight is: " << total_ngram_wgt;

    for (set<int>::const_iterator i = must_sample.begin(); i != must_sample.end(); i++) {
      map<int, BaseFloat>::const_iterator ii = bigrams.find(*i);
      if (ii != bigrams.end()) {
        total_ngram_wgt -= ii->second;
      } else {
        total_ngram_wgt -= u[*i].second * alpha;
      }
    }
    
    sort(bigram_probs.begin(), bigram_probs.end(), std::greater<BaseFloat>());

    int i = 0, j = 0;  // iteratos 2 vectors like merge sort
    // i for bigram_probs, j for u
    while (true) {
      // find the first unigram that counts
      while (j < u.size() && must_sample.find(j) != must_sample.end() && bigrams.find(j) != bigrams.end()) {
        j++;
      }
      // now neither i or j has "special" probs; both only depend on the ngram probs

      BaseFloat p;
      if (i < bigrams.size() && bigram_probs[i] > alpha * u[j].second) {
        p = bigram_probs[i];
        i++;
      } else {
        p = alpha * u[j].second;
        j++;
      }
      if (p / total_ngram_wgt * total_selection_wgt > 1.0) {
        // needs a cutoff
        total_ngram_wgt -= p;
        total_selection_wgt -= 1.0;
      } else {
        max_allowed_ngram_prob = total_ngram_wgt / total_selection_wgt;
        break;
      }
    }
//    KALDI_LOG << "max allowed ngram prob is " << max_allowed_ngram_prob;
  }

  set<int>::const_iterator must_sample_iter = must_sample.begin();
  map<int, BaseFloat>::const_iterator bigram_iter = bigrams.begin();

  for (int i = 0; i < u.size();) {
    // a must-sample word
    if (must_sample_iter != must_sample.end() && *must_sample_iter == i) {
      out->push_back(interval(i, i + 1, -1.0, ONE));
//      KALDI_LOG << "adding interval " << i << " " << i + 1 << " with selection-probs = " << 1.0;
      if (bigram_iter != bigrams.end() && *must_sample_iter == bigram_iter->first) {
        bigram_iter++;
      }
      must_sample_iter++;
//      if (bigram_iter->first == i) {
//        bigram_iter++;
//      }
      i++;
    } else if (bigram_iter != bigrams.end() && bigram_iter->first == i) {
      BaseFloat p = bigram_iter->second / max_allowed_ngram_prob;
      if (p > 1.0) {
        p = ONE;
      }
      out->push_back(interval(i, i + 1, bigram_iter->second, p));
//      KALDI_LOG << "adding interval " << i << " " << i + 1 << " with selection-probs = " << p;
      bigram_iter++;
      i++;
    } else {
      BaseFloat u_i = u[i].second * alpha;

      int n = max_allowed_ngram_prob / u_i; // since we know the original unigram is sorted we could at least have n
      int group_end = i + n;
      BaseFloat cdf_start = 0;
      if (i > 1) {
        cdf_start = cdf[i - 1];
      }
      BaseFloat current_probs = cdf[group_end - 1] - cdf_start;

      int index_upper_bound = INT_MAX;
      if (must_sample_iter != must_sample.end()) {
        index_upper_bound = std::min(index_upper_bound, *must_sample_iter);
      }
      if (bigram_iter != bigrams.end()) {
        index_upper_bound = std::min(index_upper_bound, bigram_iter->first);
      }

//      std::min(*must_sample_iter, bigram_iter->first);

//      while (group_end < u.size() && current_probs < max_allowed_ngram_prob
//                                  && group_end < index_upper_bound) {
//        n = (max_allowed_ngram_prob - current_probs) / (u[group_end].second * alpha);
//        group_end += n;
//        if (n == 0) {
//          break;
//        }
//      }
      // probably need debugging here TODO(hxu)

      group_end = std::min(group_end, int(u.size()));
      group_end = std::min(group_end, index_upper_bound);

      KALDI_ASSERT(group_end >= i);

      if (group_end == i) {
        out->push_back(interval(i, i + 1, -1, ONE));
        i++;
      } else {
        BaseFloat uni_prob = cdf[group_end - 1];
        if (i != 0) {
          uni_prob -= cdf[i - 1];
        }
        uni_prob *= alpha;
        BaseFloat selection_prob = uni_prob / max_allowed_ngram_prob;

        KALDI_ASSERT(selection_prob <= 1.0);
        out->push_back(interval(i, group_end, uni_prob, selection_prob));
  //      KALDI_LOG << "adding interval " << i << " " << group_end << " with selection-probs = " << selection_prob;
        i = group_end;
      }
    }
  }

//  CheckValidGrouping(*out, k);
  
}

// assume u is already sorted
void DoGrouping(vector<std::pair<int, BaseFloat> > u, int k, vector<interval> *out) {
  BaseFloat sum = 0.0;
  for (int i = 0; i < u.size(); i++) {
    sum += std::min(u[i].second, BaseFloat(1.0));
  }
//  for (int i = 0; i < u.size(); i++) {
//    u[i].second /= sum;
//  }

  KALDI_ASSERT(out->size() == 0);
  BaseFloat weight_to_spare = k;

  int size = u.size();

  int begin = -1;
  BaseFloat cumulative_unigram_weight = 0.0;
  BaseFloat unigram_weight_left = 1.0;
  BaseFloat selection_weight;

  using std::min;
  int i;
  for (i = 0; i < size; i++) {
    if (u[i].first == 97) {
      int j  = 0;
      j++;
    }
    if (begin == -1 && min(BaseFloat(10.0), u[i].second) / sum / unigram_weight_left * weight_to_spare >= 1) {
      // first case, the single word is too big
      weight_to_spare -= 1.0;
      unigram_weight_left -= min(BaseFloat(1.0), u[i].second) / sum;
      out->push_back(interval(i, i + 1, min(BaseFloat(1.0), u[i].second) / sum, 10.0));
    } else if (begin == -1) {
      // first time we encounter a word whose prob isn't too big
      begin = i;
      cumulative_unigram_weight = min(BaseFloat(1.0), u[i].second) / sum;
    } else {
      // in the middle of grouping words
      // first test if including the new word would make it bigger than 1
      selection_weight = (cumulative_unigram_weight + min(BaseFloat(1.0), u[i].second) / sum) / unigram_weight_left * weight_to_spare;

      if (selection_weight > 1) {
//        cout << "here, i = " << i << endl;
        // too big, then group thing from begin to i - 1
        selection_weight = cumulative_unigram_weight / unigram_weight_left * weight_to_spare;
//        weight_to_spare -= selection_weight;
//        unigram_weight_left -= cumulative_unigram_weight;
        KALDI_ASSERT(selection_weight >= 0 && selection_weight <= 1);
        out->push_back(interval(begin, i, cumulative_unigram_weight, selection_weight));

        begin = i;
        cumulative_unigram_weight = min(BaseFloat(1.0), u[i].second) / sum;
      } else {
        cumulative_unigram_weight += min(BaseFloat(1.0), u[i].second) / sum;
      }
    }
  }
  
  if (begin != -1 && begin < size) {
    selection_weight = (cumulative_unigram_weight) / unigram_weight_left * weight_to_spare;
  //  selection_weight = (cumulative_unigram_weight + min(BaseFloat(1.0), u[i].second) / sum) / unigram_weight_left * weight_to_spare;
    KALDI_ASSERT(selection_weight <= 1.0);
    out->push_back(interval(begin, i, cumulative_unigram_weight, selection_weight));
  }
//  for (int i = 0; i < out->size(); i++) {
//    const interval &j = (*out)[i];
//    cout << j.L << ", " << j.R << ": " << j.unigram_prob << ", " << j.selection_prob << endl;
//  }

  CheckValidGrouping(*out, k);
  
}

// return an int i in [L, R - 1], w/ probs porportional to their pdf's
int SelectOne(const vector<BaseFloat> &cdf, int L, int R) {
//  BaseFloat low = 0;
//  if (L - 1 >= 0) {
//    low = cdf[L];
//  }
//  BaseFloat p = RandUniform() * (cdf[R] - low) + low;
  BaseFloat p = RandUniform() * (cdf[R] - cdf[L]) + cdf[L];

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

//  cout << "select " << index << " in range " << L << "-" << R <<endl;
//  KALDI_ASSERT(index >= L && index < R);
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

unordered_map<string, int> ReadWordlist(string filename) {
  unordered_map<string, int> ans;
  ifstream ifile(filename.c_str());
  string word;
  int id;

  while (ifile >> word >> id) {
    ans[word] = id;
  }
  return ans;
}

void ReadUnigram(string f, vector<BaseFloat> *u) {
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
//    (*u)[i] = 1.0 / u->size();  // TODO(hxu)
  }
}

NnetExample GetEgsFromSent(const vector<int>& word_ids_in, int input_dim,
                           const vector<int>& word_ids_out, int output_dim) {
  SparseMatrix<BaseFloat> input_frames(word_ids_in.size(), input_dim);

  for (int j = 0; j < word_ids_in.size(); j++) {
    vector<std::pair<MatrixIndexT, BaseFloat> > pairs;
    pairs.push_back(std::make_pair(word_ids_in[j], 1.0));
    SparseVector<BaseFloat> v(input_dim, pairs);
    input_frames.SetRow(j, v);
  }

  NnetExample eg;
  eg.io.push_back(nnet3::NnetIo("input", 0, input_frames));

  Posterior posterior;
  for (int i = 0; i < word_ids_out.size(); i++) {
    vector<std::pair<int32, BaseFloat> > p;
    p.push_back(std::make_pair(word_ids_out[i], 1.0));
    posterior.push_back(p);
  }

  eg.io.push_back(nnet3::NnetIo("output", output_dim, 0, posterior));
  return eg;
}



void SampleWithoutReplacement(vector<std::pair<int, BaseFloat> > u, int n,
                              vector<int> *out) {
  sort(u.begin(), u.end(), LargerThan);
  vector<BaseFloat> cdf(u.size() + 1);
  cdf[0] = 0;
  for (int i = 1; i <= cdf.size(); i++) {
    cdf[i] = cdf[i - 1] + std::min(BaseFloat(1.0), u[i - 1].second);
  }

//    cout << "cdf: ";
//    for (int i = 0; i < cdf.size(); i++) {
//      cout << cdf[i] << " ";
//    } cout << endl;
  vector<BaseFloat> cdf2(u.size(), 0);
  cdf2[0] = u[0].second;
  for (int i = 1; i < u.size(); i++) {
    cdf2[i] = cdf2[i - 1] + u[i].second;
  }

//  KALDI_ASSERT(cdf[cdf.size() - 1], n)
  vector<interval> g;
//  DoGrouping(u, n, &g);
  DoGroupingCDF(u, cdf2, n, set<int>(), map<int, BaseFloat>(), &g);

  vector<std::pair<int, BaseFloat> > group_u(g.size());
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
    if (g[(*out)[i]].L + 1 < g[(*out)[i]].R) { // is a group of many
      int index = SelectOne(cdf, g[(*out)[i]].L, g[(*out)[i]].R);
      (*out)[i] = u[index].first;
    } else {
      (*out)[i] = u[g[(*out)[i]].L].first;
    }
  }
//  cout << "selected words are: ";
//  for (int i = 0; i < out->size(); i++) {
//    cout << (*out)[i] << " ";
//  }
//  cout << endl;
}

void SampleWithoutReplacement_(vector<std::pair<int, BaseFloat> > u, int n,
                              vector<int> *out) {
  sort(u.begin(), u.end(), LargerThan);

  KALDI_ASSERT(n != 0);

  vector<int>& ans = *out;
  ans.resize(n);

  BaseFloat tot_weight = 0;

  for (int i = 0; i < n; i++) {
    tot_weight += std::min(BaseFloat(1.0), u[i].second);
    ans[i] = i;
  }

  for (int k = n; k < u.size(); k++) {
    tot_weight += std::min(BaseFloat(1.0), u[k].second);
    BaseFloat pi_k1_k1 = u[k].second / tot_weight * n;

    if (pi_k1_k1 > 1) {
      KALDI_ASSERT(false); // never gonna happen in our setup since sorted
      pi_k1_k1 = 1;  // must add
    } else {
      BaseFloat p = RandUniform();
      if (p > pi_k1_k1) {
        continue;
      }
    }

    vector<BaseFloat> R(n);
    // fill up R
    {
      BaseFloat Lk = 0;
      BaseFloat Tk = 0;
      for (int i = 0; i < n; i++) {
        BaseFloat pi_k_i = u[ans[i]].second /
                   (tot_weight - std::min(BaseFloat(1.0), u[k].second)) * n;
        BaseFloat pi_k1_i = u[ans[i]].second / tot_weight * n;

        if (u[ans[i]].second >= 5.0) {
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

      BaseFloat sum = 0;
      for (int i = 0; i < n; i++) {
        BaseFloat pi_k_i = u[ans[i]].second /
            (tot_weight - std::min(BaseFloat(1.0), u[k].second)) * n;
        BaseFloat pi_k1_i = u[ans[i]].second / tot_weight * n;

        if (pi_k_i < 1 && pi_k1_i < 1) {
          // case C
          R[i] = (1 - Tk) / (n - Lk);
        }
        sum += R[i];
      }
      KALDI_ASSERT(ApproxEqual(sum, 1.0));
    }

    vector<BaseFloat> cdf(R);
    BaseFloat *cdf_ptr = &(cdf[0]);
    double cdf_sum = 0.0;
    for (int32 i = 0, size = cdf.size(); i < size; i++) {
      cdf_sum += cdf_ptr[i];
      cdf_ptr[i] = static_cast<BaseFloat>(cdf_sum);
    }

    BaseFloat p = RandUniform() * cdf[cdf.size() - 1];
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
    ans[index] = k;

//    bool replaced = false;
//    for (int i = 0; i < n; i++) {
//      p -= R[i];
//      if (p <= 0) {
//        ans[i] = k;
//        KALDI_ASSERT(abs(i - index) < 2);
//        replaced = true;
//        break;
//      }
//    }
//
//    if (!replaced) {
//      KALDI_LOG << "p should be close to 0; it is " << p;
//      for (int i = 1; ; i++) {
//        if (u[ans[n - i]].second < 1) {
//          ans[n - 1] = k;
//          break;
//        }
//      }
//    }
  }

  //  change to the correct indexes
  for (int i = 0; i < ans.size(); i++) {
    ans[i] = u[ans[i]].first;
  }
}

void NormalizeVec(int k, const set<int>& ones, vector<BaseFloat> *probs) {
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
      BaseFloat &t = (*probs)[i];
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
      BaseFloat &t = (*probs)[i];
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

//  KALDI_ASSERT(nnet1.NumComponents() == nnet2.NumComponents());
//  int32 updatable_c = 0;
//  for (int32 c = 0; c < nnet1.NumComponents(); c++) {
//    const Component *comp1 = nnet1.GetComponent(c),
//                    *comp2 = nnet2.GetComponent(c);
//    if (comp1->Properties() & kUpdatableComponent) {
//      const UpdatableComponent
//          *u_comp1 = dynamic_cast<const UpdatableComponent*>(comp1),
//          *u_comp2 = dynamic_cast<const UpdatableComponent*>(comp2);
//      KALDI_ASSERT(u_comp1 != NULL && u_comp2 != NULL);
//      dot_prod->Data()[updatable_c] = u_comp1->DotProduct(*u_comp2);
//      updatable_c++;
//    }
//  }
//  KALDI_ASSERT(updatable_c == dot_prod->Dim());

  int32 dim = dot_prod->Dim();
  dot_prod->Data()[dim - 2] = nnet1.I()->DotProduct(*nnet2.I());
  dot_prod->Data()[dim - 1] = nnet1.O()->DotProduct(*nnet2.O());
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
