#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

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
  
//  set the 1s to be 10 to avoid numerical issues
  for (set<int>::const_iterator iter = ones.begin(); iter != ones.end(); iter++) {
    sum -= (*probs)[*iter];
    (*probs)[*iter] = 10;  // mark the ones
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
