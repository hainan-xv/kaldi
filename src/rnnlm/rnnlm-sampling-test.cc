#include <math.h>

#include "rnnlm-training.h"
#include "cudamatrix/cu-matrix.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {
namespace rnnlm {

void UnitTestSamplingNonlinearity() {
  for (int i = 0; i < 100; i++) {
    int num_rows = rand() % 200 + 200;
    int num_cols = rand() % 200 + 200;

    // forward pass...
    CuMatrix<BaseFloat> in1(num_rows, num_cols);
    CuMatrix<BaseFloat> out1(num_rows, num_cols);
    in1.SetRandn();
    in1.Add(-1.0);
    ComputeSamplingNonlinearity(in1, &out1);

    // testing the forward non-linearity
    Matrix<BaseFloat> in2(in1);
    Matrix<BaseFloat> out2(num_rows, num_cols);
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols; j++) {
        BaseFloat x = in2(i, j);
        if (x <= 0) {
          out2(i, j) = exp(x);
        } else {
          out2(i, j) = 1 + x;
        }
      }
    }

    CuMatrix<BaseFloat> tmp(out2);
    KALDI_ASSERT(tmp.ApproxEqual(out1));

    // compute objf
    CuVector<BaseFloat> probs(num_cols);
    probs.SetRandUniform();
    probs.Scale(0.5);
    probs.Add(0.5);
    probs.InvertElements();
    probs.Scale(-1);

    BaseFloat objf = 0.0;
    tmp.CopyRowsFromVec(probs);
    tmp.MulElements(out1);
    // now each element of tmp is -f(y)/prob(y)

    objf = tmp.Sum();
    KALDI_LOG << "original objf is " << objf;

    // backward
    CuMatrix<BaseFloat> in_deriv(num_rows, num_cols);
    BackpropSamplingNonlinearity(probs, &out1, &in_deriv);

    // objf with delta
    CuMatrix<BaseFloat> new_in(num_rows, num_cols);
    CuMatrix<BaseFloat> new_out(num_rows, num_cols);
    int test_dim = 3;

    Vector<BaseFloat> measured_objf_change(test_dim),
                      predicted_objf_change(test_dim);
    for (int t = 0; t < test_dim; t++) {
      CuMatrix<BaseFloat> delta_in(num_rows, num_cols);
      delta_in.SetRandn();
      delta_in.Scale(0.001);

      new_in.CopyFromMat(in1);
      new_in.AddMat(1.0, delta_in);

      ComputeSamplingNonlinearity(new_in, &new_out);

      BaseFloat new_objf = 0.0;
      tmp.CopyRowsFromVec(probs);
      tmp.MulElements(new_out);

      new_objf = tmp.Sum();
//      KALDI_LOG << "new objf is " << new_objf;
//      KALDI_LOG << "delta objf is " << new_objf - objf;

      BaseFloat measured_change = new_objf - objf;
      BaseFloat predicted_change = TraceMatMat(in_deriv, delta_in, kTrans);

      measured_objf_change(t) = measured_change;
      predicted_objf_change(t) = predicted_change;
//      KALDI_LOG << "predicted change of objf is " << predicted_change;
//      KALDI_LOG << "";
    }
    KALDI_LOG << "Predicted objf-change = " << predicted_objf_change;
    KALDI_LOG << "Measured objf-change = " << measured_objf_change;
    BaseFloat threshold = 0.1;

    bool ans = ApproxEqual(predicted_objf_change, measured_objf_change,
                           threshold);
    if (!ans) {
      KALDI_WARN << "Model-derivative test failed";
    }
  }
}
}
}

int main() {
  using namespace kaldi;
  using namespace rnnlm;


  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    UnitTestSamplingNonlinearity();

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
  SetVerboseLevel(4);
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;

}
