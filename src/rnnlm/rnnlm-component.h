// nnet3/nnet-simple-component.h


// See ../../COPYING for clarification regarding multiple authors
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
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_RNNLM_RNNLM_COMPONENT_H_
#define KALDI_RNNLM_RNNLM_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "rnnlm/rnnlm-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "rnnlm/nnet-parse.h"
#include <iostream>

namespace kaldi {
namespace rnnlm {

class LmSoftmaxComponent;
class LmLogSoftmaxComponent;

using std::vector;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
class LmLinearComponent: public LmInputComponent {
  friend class LmSoftmaxComponent; // Friend declaration relates to mixing up.
 public:

  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  LmLinearComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "LmLinearComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }

  virtual void Backprop(const SparseMatrix<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmInputComponent *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv = NULL) const;

  virtual void Propagate(const SparseMatrix<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmInputComponent* Copy() const;

  // Some functions from base-class LmUpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmInputComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmInputComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(//const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
//  const Vector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit LmLinearComponent(const LmLinearComponent &other);
  // The next constructor is used in converting from nnet1.
  LmLinearComponent(const MatrixBase<BaseFloat> &linear_params,
//                  const VectorBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev);//, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

 protected:
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }

  virtual void Update(
      const SparseMatrix<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }

  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  virtual void UpdateSimple(
      const SparseMatrix<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const LmLinearComponent &operator = (const LmLinearComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
};

class LmNaturalGradientLinearComponent: public LmLinearComponent {
 public:
  void FreezeNaturalGradient(bool freeze);
  virtual std::string Type() const { return "LmNaturalGradientLinearComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);
  void Init(int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);
  // this constructor does not really initialize, use Init() or Read().
  LmNaturalGradientLinearComponent();
  virtual void Resize(int32 input_dim, int32 output_dim);
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Info() const;
  virtual LmInputComponent* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmInputComponent &other);
  // copy constructor
  explicit LmNaturalGradientLinearComponent(
      const LmNaturalGradientLinearComponent &other);
  virtual void ZeroStats();

 private:
  // disallow assignment operator.
  LmNaturalGradientLinearComponent &operator= (
      const LmNaturalGradientLinearComponent&);

  // Configs for preconditioner.  The input side tends to be better conditioned ->
  // smaller rank needed, so make them separately configurable.
  int32 rank_in_;
  int32 rank_out_;
  int32 update_period_;
  BaseFloat num_samples_history_;
  BaseFloat alpha_;

  nnet3::OnlineNaturalGradient preconditioner_in_;

  nnet3::OnlineNaturalGradient preconditioner_out_;

  // If > 0, max_change_per_sample_ is the maximum amount of parameter
  // change (in L2 norm) that we allow per sample, averaged over the minibatch.
  // This was introduced in order to control instability.
  // Instead of the exact L2 parameter change, for
  // efficiency purposes we limit a bound on the exact
  // change.  The limit is applied via a constant <= 1.0
  // for each minibatch, A suitable value might be, for
  // example, 10 or so; larger if there are more
  // parameters.
  BaseFloat max_change_per_sample_;

  // update_count_ records how many updates we have done.
  double update_count_;

  // active_scaling_count_ records how many updates we have done,
  // where the scaling factor is active (not 1.0).
  double active_scaling_count_;

  // max_change_scale_stats_ records the sum of scaling factors
  // in each update, so we can compute the averaged scaling factor
  // in Info().
  double max_change_scale_stats_;

  // Sets the configs rank, alpha and eta in the preconditioner objects,
  // from the class variables.
  void SetNaturalGradientConfigs();

  virtual void Update(
      const SparseMatrix<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);
};

class AffineImportanceSamplingComponent: public LmOutputComponent {
 public:
  virtual int32 InputDim() const { return params_.NumCols() - 1; }
  virtual int32 OutputDim() const { return params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  AffineImportanceSamplingComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineImportanceSamplingComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }


  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         const vector<int> &indexes,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         bool normalize,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual BaseFloat ComputeLogprobOfWordGivenHistory(const CuVectorBase<BaseFloat> &hidden,
                                                     int32 word_index) const;

  virtual void Backprop(const vector<int> &indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmOutputComponent *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Backprop(
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmOutputComponent *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmOutputComponent* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmOutputComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmOutputComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  void SetParams(const CuMatrixBase<BaseFloat> &params);

  explicit AffineImportanceSamplingComponent(const AffineImportanceSamplingComponent &other);

  AffineImportanceSamplingComponent(const CuMatrixBase<BaseFloat> &params,
                                    BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.

 protected:

  const AffineImportanceSamplingComponent &operator = (const AffineImportanceSamplingComponent &other); // Disallow.
  CuMatrix<BaseFloat> params_;
};


class NaturalGradientAffineImportanceSamplingComponent: public AffineImportanceSamplingComponent {
 public:
  void FreezeNaturalGradient(bool freeze);

  virtual int32 InputDim() const { return params_.NumCols() - 1; }
  virtual int32 OutputDim() const { return params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  NaturalGradientAffineImportanceSamplingComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "NaturalGradientAffineImportanceSamplingComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }


  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         const vector<int> &indexes,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         bool normalize,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const vector<int> &indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmOutputComponent *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Backprop(
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmOutputComponent *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  void SetNaturalGradientConfigs();

  virtual LmOutputComponent* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmOutputComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmOutputComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  void SetParams(const CuMatrixBase<BaseFloat> &params);

  explicit NaturalGradientAffineImportanceSamplingComponent(const NaturalGradientAffineImportanceSamplingComponent &other);

  NaturalGradientAffineImportanceSamplingComponent(const CuMatrixBase<BaseFloat> &params,
                                    BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);
  void Init(int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);

  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.

 protected:

  const NaturalGradientAffineImportanceSamplingComponent &operator = (const NaturalGradientAffineImportanceSamplingComponent &other); // Disallow.
//  CuMatrix<BaseFloat> params_;

  int32 rank_in_;
  int32 rank_out_;
  int32 update_period_;
  BaseFloat num_samples_history_;
  BaseFloat alpha_;
  nnet3::OnlineNaturalGradient preconditioner_in_;

  nnet3::OnlineNaturalGradient preconditioner_out_;

  BaseFloat max_change_per_sample_;

  // update_count_ records how many updates we have done.
  double update_count_;

  // active_scaling_count_ records how many updates we have done,
  // where the scaling factor is active (not 1.0).
  double active_scaling_count_;

  // max_change_scale_stats_ records the sum of scaling factors
  // in each update, so we can compute the averaged scaling factor
  // in Info().
  double max_change_scale_stats_;
};


} // namespace rnnlm
} // namespace kaldi


#endif
