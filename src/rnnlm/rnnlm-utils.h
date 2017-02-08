#ifndef KALDI_RNNLM_UTILS_H_
#define KALDI_RNNLM_UTILS_H_

#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-diagnostics.h"

#include "rnnlm/rnnlm-nnet.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>


using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::set;

namespace kaldi {
namespace rnnlm {

using nnet3::NnetExample;

const int kOosId = 1;

// convert a vector of integers v to a SparseMatrx m such that
// v.size() == m.NumRows()
// and m(i, j) == 1 iff v[i] == j
void VectorToSparseMatrix(const vector<int32> &v,
                          int dim,
                          SparseMatrix<BaseFloat> *sp);

// the reverse operation of VectorToSparseMatrix, i.e.
// convert a SparseMatrx which we know each row has exactly one non-zero number
// and convert that to a vector of integer
// i.e. m(i, j) != 0 iff v[i] == j
void SparseMatrixToVector(const SparseMatrix<BaseFloat> &sp,
                          vector<int32> *v);

vector<string> SplitByWhiteSpace(const string &line);

unordered_map<string, int> ReadWordlist(string filename);

NnetExample GetEgsFromSent(const vector<int>& word_ids_in, int input_dim,
                           const vector<int>& word_ids_out, int output_dim);

// u should be the result of calling NormalizeVec(), i.e.
// it should add up to n
void SampleWithoutReplacement(vector<std::pair<int, BaseFloat> > u, int n, vector<int> *out);

// normalize the prob vector such that
// every prob satisfies 0 < p <= 1
// sum of all probs equal k
// in particular, probs[i] must be 1 if i is in the set "ones"
//  *** one thing to know is, to avoid numerical issues, use value 10.0 to
// represent prob = 1; so in the code that operates on the vector
// sometimes we will see min(1.0, probs[i])

// probs should add up to 1 initually (a valid unigram distribution)
void NormalizeVec(int k, const set<int> &ones, vector<BaseFloat> *probs);

bool LargerThan(const std::pair<int, BaseFloat> &t1,
                const std::pair<int, BaseFloat> &t2);

void ReadUnigram(string f, vector<BaseFloat> *u);

void ComponentDotProducts(const LmNnet &nnet1,
                          const LmNnet &nnet2,
                          VectorBase<BaseFloat> *dot_prod);

std::string PrintVectorPerUpdatableComponent(const LmNnet &nnet,
                                             const VectorBase<BaseFloat> &vec);

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_RNNLM_UTILS_H_
