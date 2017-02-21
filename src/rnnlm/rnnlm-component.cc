
#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "rnnlm/nnet-parse.h"
#include "rnnlm/rnnlm-component.h"
#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

void NaturalGradientAffineImportanceSamplingComponent::Scale(BaseFloat scale) {
  params_.Scale(scale);
}

void NaturalGradientAffineImportanceSamplingComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  params_.Resize(output_dim, input_dim + 1);
}

void NaturalGradientAffineImportanceSamplingComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const NaturalGradientAffineImportanceSamplingComponent *other =
             dynamic_cast<const NaturalGradientAffineImportanceSamplingComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

NaturalGradientAffineImportanceSamplingComponent::NaturalGradientAffineImportanceSamplingComponent(
                            const NaturalGradientAffineImportanceSamplingComponent &component):
    AffineImportanceSamplingComponent(component)
//    params_(component.params_)
    { }

//NaturalGradientAffineImportanceSamplingComponent::NaturalGradientAffineImportanceSamplingComponent(
//                                   const CuMatrixBase<BaseFloat> &params,
//                                   BaseFloat learning_rate):
//                                            params_(params) {
//  SetUnderlyingLearningRate(learning_rate);
//}

void NaturalGradientAffineImportanceSamplingComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  params_.SetZero();
}

void NaturalGradientAffineImportanceSamplingComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &params) {
  params_ = params;
}

void NaturalGradientAffineImportanceSamplingComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}

std::string NaturalGradientAffineImportanceSamplingComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  nnet3::PrintParameterStats(stream, "params", params_);
  return stream.str();
}

LmComponent* NaturalGradientAffineImportanceSamplingComponent::Copy() const {
  NaturalGradientAffineImportanceSamplingComponent *ans = new NaturalGradientAffineImportanceSamplingComponent(*this);
  return ans;
}

BaseFloat NaturalGradientAffineImportanceSamplingComponent::DotProduct(const LmComponent &other_in) const {
  const NaturalGradientAffineImportanceSamplingComponent *other =
      dynamic_cast<const NaturalGradientAffineImportanceSamplingComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(params_, other->params_, kTrans);
}

void NaturalGradientAffineImportanceSamplingComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  params_.Resize(output_dim, input_dim + 1);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  params_.SetRandn(); // sets to random normally distributed noise.
  params_.ColRange(0, input_dim).Scale(param_stddev);
  params_.ColRange(input_dim, 1).Scale(bias_stddev);
}

void NaturalGradientAffineImportanceSamplingComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &params_); // will abort on failure.
}

void NaturalGradientAffineImportanceSamplingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
//    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
//        bias_stddev = 1.0;
    BaseFloat param_stddev = 1.0, /// log(1.0 / output_dim),
        bias_stddev = log(1.0 / output_dim);

    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);

    bias_stddev = log(1.0 / output_dim);

    Init(input_dim, output_dim, param_stddev, bias_stddev);

    // TODO(hxu)
    params_.ColRange(params_.NumCols() - 1, 1).Set(bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// const BaseFloat kCutoff = 1.0;

void NaturalGradientAffineImportanceSamplingComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  CuSubMatrix<BaseFloat> bias_params_sub(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  KALDI_ASSERT(out->NumRows() == in.NumRows());
  CuMatrix<BaseFloat> new_linear(indexes.size(), params_.NumCols() - 1);
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params, idx);

  CuMatrix<BaseFloat> bias_params(1, params_.NumRows());
  bias_params.AddMat(1.0, bias_params_sub, kTrans);

  // BUGGY here TODO(hxu)
  out->RowRange(0, 1).AddCols(bias_params, idx);
  out->CopyRowsFromVec(out->Row(0));
  out->AddMatMat(1.0, in, kNoTrans, new_linear, kTrans, 1.0);
}

void NaturalGradientAffineImportanceSamplingComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                  bool normalize,
                                                  CuMatrixBase<BaseFloat> *out) const {
  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));
  out->Row(0).CopyColFromMat(bias_params, 0);
  out->CopyRowsFromVec(out->Row(0));
  out->AddMatMat(1.0, in, kNoTrans, linear_params, kTrans, 1.0);
  if (normalize) {
    out->ApplyLogSoftMaxPerRow(*out);
  }
}

void NaturalGradientAffineImportanceSamplingComponent::Backprop(
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  CuMatrix<BaseFloat> tmp(out_value);
  tmp.Set(0.0);
  tmp.Row(0).CopyColFromMat(bias_params, 0);
  tmp.CopyRowsFromVec(tmp.Row(0));
  tmp.AddMatMat(1.0, in_value, kNoTrans, linear_params, kTrans, 1.0);

  // now tmp is the in_value for log-softmax

  tmp.DiffLogSoftmaxPerRow(tmp, output_deriv);

  if (input_deriv != NULL)
    input_deriv->AddMatMat(1.0, output_deriv, kNoTrans, linear_params, kNoTrans,
                           1.0);

  NaturalGradientAffineImportanceSamplingComponent* to_update
             = dynamic_cast<NaturalGradientAffineImportanceSamplingComponent*>(to_update_0);

  if (to_update != NULL) {

    // TODO(hxu) need to add natural gradient


    CuMatrix<BaseFloat> delta(1, params_.NumRows(), kSetZero);
    delta.Row(0).AddRowSumMat(learning_rate_, output_deriv, 1.0);
    to_update->params_.ColRange(params_.NumCols() - 1, 1).AddMat(1.0, delta, kTrans);
    to_update->params_.ColRange(0, params_.NumCols() - 1).AddMatMat(learning_rate_, output_deriv, kTrans,
                                in_value, kNoTrans, 1.0);
  }
}

void NaturalGradientAffineImportanceSamplingComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  const CuMatrixBase<BaseFloat> &new_out_deriv = output_deriv;

  CuMatrix<BaseFloat> new_linear(indexes.size(), linear_params.NumCols());
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params, idx);

  input_deriv->AddMatMat(1.0, new_out_deriv, kNoTrans, new_linear, kNoTrans, 1.0);

  NaturalGradientAffineImportanceSamplingComponent* to_update
             = dynamic_cast<NaturalGradientAffineImportanceSamplingComponent*>(to_update_0);

  if (to_update != NULL) {

    CuMatrix<BaseFloat> delta_bias(1, output_deriv.NumCols(), kSetZero);
    new_linear.SetZero();  // clear the contents

    {
      KALDI_LOG << "here";
      CuMatrix<BaseFloat> in_value_temp;
      in_value_temp.Resize(in_value.NumRows(),
                           in_value.NumCols() + 1, kUndefined);
      in_value_temp.Range(0, in_value.NumRows(),
                          0, in_value.NumCols()).CopyFromMat(in_value);

      // Add the 1.0 at the end of each row "in_value_temp"
      in_value_temp.Range(0, in_value.NumRows(),
                          in_value.NumCols(), 1).Set(1.0);

      CuMatrix<BaseFloat> out_deriv_temp(output_deriv);

      CuMatrix<BaseFloat> row_products(2, in_value.NumRows());
      CuSubVector<BaseFloat> in_row_products(row_products, 0),
          out_row_products(row_products, 1);

      // These "scale" values get will get multiplied into the learning rate (faster
      // than having the matrices scaled inside the preconditioning code).
      BaseFloat in_scale, out_scale;

      to_update->preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                                &in_scale);
      to_update->preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                                 &out_scale);

      // "scale" is a scaling factor coming from the PreconditionDirections calls
      // (it's faster to have them output a scaling factor than to have them scale
      // their outputs).
      BaseFloat scale = in_scale * out_scale;

      CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                                  0, in_value_temp.NumRows(),
                                                  0, in_value_temp.NumCols() - 1);
      // this "precon_ones" is what happens to the vector of 1's representing
      // offsets, after multiplication by the preconditioner.
      CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

      precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

      BaseFloat local_lrate = scale * learning_rate_;
      to_update->update_count_ += 1.0;
      delta_bias.Row(0).AddMatVec(local_lrate, out_deriv_temp, kTrans,
                           precon_ones, 1.0);
      new_linear.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
    }

//    new_linear.AddMatMat(learning_rate_, new_out_deriv, kTrans,
//                         in_value, kNoTrans, 1.0);
//    delta_bias.Row(0).AddRowSumMat(learning_rate_, new_out_deriv, kTrans);

    vector<int> indexes_2(bias_params.NumRows(), -1);
    for (int i = 0; i < indexes.size(); i++) {
      indexes_2[indexes[i]] = i;
    }

    CuArray<int> idx2(indexes_2);
    to_update->params_.ColRange(0, params_.NumCols() - 1).AddRows(1.0, new_linear, idx2);

    CuMatrix<BaseFloat> delta_bias_trans(output_deriv.NumCols(), 1, kSetZero);
    delta_bias_trans.AddMat(1.0, delta_bias, kTrans);

    to_update->params_.ColRange(params_.NumCols() - 1, 1).AddRows(1.0, delta_bias_trans, idx2);  // TODO(hxu)

    BaseFloat t = TraceMatMat(to_update->params_, to_update->params_, kTrans);

//    KALDI_LOG << "tracematmat on to_update out is " << t;

  }
}

void NaturalGradientAffineImportanceSamplingComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</NaturalGradientAffineImportanceSamplingComponent>");
}

void NaturalGradientAffineImportanceSamplingComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</NaturalGradientAffineImportanceSamplingComponent>");
}

int32 NaturalGradientAffineImportanceSamplingComponent::NumParameters() const {
  return (InputDim() ) * ( 1 + OutputDim());
}

void NaturalGradientAffineImportanceSamplingComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyRowsFromMat(params_);
}
void NaturalGradientAffineImportanceSamplingComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
//  KALDI_ASSERT(params.Dim() == this->NumParameters());
  params_.CopyRowsFromVec(params);
//  bias_params_.Row(0).CopyFromVec(params.Range(InputDim() * OutputDim(),
//                                        OutputDim()));
}


LmNaturalGradientLinearComponent::LmNaturalGradientLinearComponent(
    const LmNaturalGradientLinearComponent &other):
    LmLinearComponent(other),
    rank_in_(other.rank_in_),
    rank_out_(other.rank_out_),
    update_period_(other.update_period_),
    num_samples_history_(other.num_samples_history_),
    alpha_(other.alpha_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_),
    max_change_per_sample_(other.max_change_per_sample_),
    update_count_(other.update_count_),
    active_scaling_count_(other.active_scaling_count_),
    max_change_scale_stats_(other.max_change_scale_stats_) {
  SetNaturalGradientConfigs();
}

void LmNaturalGradientLinearComponent::Update(
    const SparseMatrix<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;

//  in_value_temp.Resize(in_value.NumRows(),
//                       in_value.NumCols() + 1, kUndefined);
////  in_value_temp.Range(0, in_value.NumRows(),
////                      0, in_value.NumCols()).CopyFromMat(in_value);
////  CuMatrixBase<BaseFloat> &m = in_value_temp.Range(0, in_value.NumRows(), 0, in_value.NumCols());
//  CuSubMatrix<BaseFloat> m(in_value_temp.Data(), in_value.NumRows(), in_value.NumCols(), in_value_temp.Stride());
//  in_value.CopyToMat(&m);
//
//  // Add the 1.0 at the end of each row "in_value_temp"
//  in_value_temp.Range(0, in_value.NumRows(),
//                      in_value.NumCols(), 1).Set(1.0);

  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  CuMatrix<BaseFloat> row_products(2,
                                   in_value.NumRows());
  CuSubVector<BaseFloat> in_row_products(row_products, 0),
      out_row_products(row_products, 1);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale = 1.0, out_scale;

//  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
//                                            &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                             &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;

//  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
//                                              0, in_value_temp.NumRows(),
//                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
//  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

//  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;
  update_count_ += 1.0;
//  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
//                         precon_ones, 1.0);

//  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
//                           in_value_precon_part, kNoTrans, 1.0);
  cu::UpdateSimpleAffineOnSparse(local_lrate, out_deriv_temp, in_value, &linear_params_);
}

void LmNaturalGradientLinearComponent::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;

  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);

  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  CuMatrix<BaseFloat> row_products(2,
                                   in_value.NumRows());
  CuSubVector<BaseFloat> in_row_products(row_products, 0),
      out_row_products(row_products, 1);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;

  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                            &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                             &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;

  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;
  update_count_ += 1.0;
//  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
//                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void LmNaturalGradientLinearComponent::ZeroStats()  {
  update_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
  active_scaling_count_ = 0.0;
}

void LmNaturalGradientLinearComponent::Scale(BaseFloat scale) {
  update_count_ *= scale;
  max_change_scale_stats_ *= scale;
  active_scaling_count_ *= scale;
  linear_params_.Scale(scale);
}

void LmNaturalGradientLinearComponent::Add(BaseFloat alpha, const LmInputComponent &other_in) {
  const LmNaturalGradientLinearComponent *other =
      dynamic_cast<const LmNaturalGradientLinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  update_count_ += alpha * other->update_count_;
  max_change_scale_stats_ += alpha * other->max_change_scale_stats_;
  active_scaling_count_ += alpha * other->active_scaling_count_;
  linear_params_.AddMat(alpha, other->linear_params_);
}

LmNaturalGradientLinearComponent::LmNaturalGradientLinearComponent():
    max_change_per_sample_(0.0),
    update_count_(0.0), active_scaling_count_(0.0),
    max_change_scale_stats_(0.0) { }

// virtual
void LmNaturalGradientLinearComponent::Resize(
    int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 1 && output_dim > 1);
  if (rank_in_ >= input_dim) rank_in_ = input_dim - 1;
  if (rank_out_ >= output_dim) rank_out_ = output_dim - 1;
  linear_params_.Resize(output_dim, input_dim);
  nnet3::OnlineNaturalGradient temp;
  preconditioner_in_ = temp;
  preconditioner_out_ = temp;
  SetNaturalGradientConfigs();
}


void LmNaturalGradientLinearComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in_);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out_);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period_);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<UpdateCount>") {
    ReadBasicType(is, binary, &update_count_);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &active_scaling_count_);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &max_change_scale_stats_);
    ReadToken(is, binary, &token);
  }
  if (token != "<LmNaturalGradientLinearComponent>" &&
      token != "</LmNaturalGradientLinearComponent>")
    KALDI_ERR << "Expected <LmNaturalGradientLinearComponent> or "
              << "</LmNaturalGradientLinearComponent>, got " << token;
  SetNaturalGradientConfigs();
}

void LmNaturalGradientLinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.0;
  int32 input_dim = -1, output_dim = -1, rank_in = 20, rank_out = 80,
      update_period = 4;
  InitLearningRatesFromConfig(cfl);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-sample", &max_change_per_sample);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample,
         matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0, bias_mean = 0.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
    Init(input_dim, output_dim, param_stddev,
         rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LmNaturalGradientLinearComponent::SetNaturalGradientConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}

void LmNaturalGradientLinearComponent::Init(
    int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void LmNaturalGradientLinearComponent::Init(
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  if (max_change_per_sample > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_sample for "
               << "LmNaturalGradientLinearComponent. But it has been deprecated. "
               << "Please use max_change for all updatable components instead "
               << "to activate the per-component max change mechanism.";
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void LmNaturalGradientLinearComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, rank_in_);
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, rank_out_);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period_);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChangePerSample>");
  WriteBasicType(os, binary, max_change_per_sample_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<UpdateCount>");
  WriteBasicType(os, binary, update_count_);
  WriteToken(os, binary, "<ActiveScalingCount>");
  WriteBasicType(os, binary, active_scaling_count_);
  WriteToken(os, binary, "<MaxChangeScaleStats>");
  WriteBasicType(os, binary, max_change_scale_stats_);
  WriteToken(os, binary, "</LmNaturalGradientLinearComponent>");
}

std::string LmNaturalGradientLinearComponent::Info() const {
  std::ostringstream stream;
  stream << LmInputComponent::Info();
  nnet3::PrintParameterStats(stream, "linear-params", linear_params_);
  stream << ", rank-in=" << rank_in_
         << ", rank-out=" << rank_out_
         << ", num_samples_history=" << num_samples_history_
         << ", update_period=" << update_period_
         << ", alpha=" << alpha_
         << ", max-change-per-sample=" << max_change_per_sample_;
  if (update_count_ > 0.0 && max_change_per_sample_ > 0.0) {
    stream << ", avg-scaling-factor=" << max_change_scale_stats_ / update_count_
           << ", active-scaling-portion="
           << active_scaling_count_ / update_count_;
  }
  return stream.str();
}

LmInputComponent* LmNaturalGradientLinearComponent::Copy() const {
  return new LmNaturalGradientLinearComponent(*this);
}

void AffineImportanceSamplingComponent::Scale(BaseFloat scale) {
  params_.Scale(scale);
}

void AffineImportanceSamplingComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  params_.Resize(output_dim, input_dim + 1);
}

void AffineImportanceSamplingComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const AffineImportanceSamplingComponent *other =
             dynamic_cast<const AffineImportanceSamplingComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

AffineImportanceSamplingComponent::AffineImportanceSamplingComponent(
                            const AffineImportanceSamplingComponent &component):
    LmOutputComponent(component),
    params_(component.params_)
    { }

AffineImportanceSamplingComponent::AffineImportanceSamplingComponent(
                                   const CuMatrixBase<BaseFloat> &params,
                                   BaseFloat learning_rate):
                                            params_(params) {
  SetUnderlyingLearningRate(learning_rate);
}

void AffineImportanceSamplingComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  params_.SetZero();
}

void AffineImportanceSamplingComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &params) {
  params_ = params;
}

void AffineImportanceSamplingComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}

std::string AffineImportanceSamplingComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  nnet3::PrintParameterStats(stream, "params", params_);
  return stream.str();
}

LmComponent* AffineImportanceSamplingComponent::Copy() const {
  AffineImportanceSamplingComponent *ans = new AffineImportanceSamplingComponent(*this);
  return ans;
}

BaseFloat AffineImportanceSamplingComponent::DotProduct(const LmComponent &other_in) const {
  const AffineImportanceSamplingComponent *other =
      dynamic_cast<const AffineImportanceSamplingComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(params_, other->params_, kTrans);
}

void AffineImportanceSamplingComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  params_.Resize(output_dim, input_dim + 1);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  params_.SetRandn(); // sets to random normally distributed noise.
  params_.ColRange(0, input_dim).Scale(param_stddev);
  params_.ColRange(input_dim, 1).Scale(bias_stddev);
}

void AffineImportanceSamplingComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &params_); // will abort on failure.
}

void AffineImportanceSamplingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
//    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
//        bias_stddev = 1.0;
    BaseFloat param_stddev = 0.0, /// log(1.0 / output_dim),
        bias_stddev = log(1.0 / output_dim);

    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);

    bias_stddev = log(1.0 / output_dim);

    Init(input_dim, output_dim, param_stddev, bias_stddev);

    // TODO(hxu)
    params_.ColRange(params_.NumCols() - 1, 1).Set(bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// const BaseFloat kCutoff = 1.0;

void AffineImportanceSamplingComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  CuSubMatrix<BaseFloat> bias_params_sub(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  KALDI_ASSERT(out->NumRows() == in.NumRows());
  CuMatrix<BaseFloat> new_linear(indexes.size(), params_.NumCols() - 1);
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params, idx);

  CuMatrix<BaseFloat> bias_params(1, params_.NumRows());
  bias_params.AddMat(1.0, bias_params_sub, kTrans);

  // BUGGY here TODO(hxu)
  out->RowRange(0, 1).AddCols(bias_params, idx);
  out->CopyRowsFromVec(out->Row(0));
  out->AddMatMat(1.0, in, kNoTrans, new_linear, kTrans, 1.0);
}

void AffineImportanceSamplingComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                  bool normalize,
                                                  CuMatrixBase<BaseFloat> *out) const {
  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));
  out->Row(0).CopyColFromMat(bias_params, 0);
  out->CopyRowsFromVec(out->Row(0));
  out->AddMatMat(1.0, in, kNoTrans, linear_params, kTrans, 1.0);
  if (normalize) {
    out->ApplyLogSoftMaxPerRow(*out);
  }
}

void AffineImportanceSamplingComponent::Backprop(
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  CuMatrix<BaseFloat> tmp(out_value);
  tmp.Set(0.0);
  tmp.Row(0).CopyColFromMat(bias_params, 0);
  tmp.CopyRowsFromVec(tmp.Row(0));
  tmp.AddMatMat(1.0, in_value, kNoTrans, linear_params, kTrans, 1.0);

  // now tmp is the in_value for log-softmax

  tmp.DiffLogSoftmaxPerRow(tmp, output_deriv);

  if (input_deriv != NULL)
    input_deriv->AddMatMat(1.0, output_deriv, kNoTrans, linear_params, kNoTrans,
                           1.0);

  AffineImportanceSamplingComponent* to_update
             = dynamic_cast<AffineImportanceSamplingComponent*>(to_update_0);

  if (to_update != NULL) {
    CuMatrix<BaseFloat> delta(1, params_.NumRows(), kSetZero);
    delta.Row(0).AddRowSumMat(learning_rate_, output_deriv, 1.0);
    to_update->params_.ColRange(params_.NumCols() - 1, 1).AddMat(1.0, delta, kTrans);
    to_update->params_.ColRange(0, params_.NumCols() - 1).AddMatMat(learning_rate_, output_deriv, kTrans,
                                in_value, kNoTrans, 1.0);
  }
}

void AffineImportanceSamplingComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  CuSubMatrix<BaseFloat> bias_params(params_.ColRange(params_.NumCols() - 1, 1));
  CuSubMatrix<BaseFloat> linear_params(params_.ColRange(0, params_.NumCols() - 1));

  const CuMatrixBase<BaseFloat> &new_out_deriv = output_deriv;

  CuMatrix<BaseFloat> new_linear(indexes.size(), linear_params.NumCols());
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params, idx);

  input_deriv->AddMatMat(1.0, new_out_deriv, kNoTrans, new_linear, kNoTrans, 1.0);

  AffineImportanceSamplingComponent* to_update
             = dynamic_cast<AffineImportanceSamplingComponent*>(to_update_0);

  if (to_update != NULL) {
    new_linear.SetZero();  // clear the contents
    new_linear.AddMatMat(learning_rate_, new_out_deriv, kTrans,
                         in_value, kNoTrans, 1.0);
    CuMatrix<BaseFloat> delta_bias(1, output_deriv.NumCols(), kSetZero);
    delta_bias.Row(0).AddRowSumMat(learning_rate_, new_out_deriv, kTrans);

    vector<int> indexes_2(bias_params.NumRows(), -1);
    for (int i = 0; i < indexes.size(); i++) {
      indexes_2[indexes[i]] = i;
    }

    CuArray<int> idx2(indexes_2);
    to_update->params_.ColRange(0, params_.NumCols() - 1).AddRows(1.0, new_linear, idx2);

    CuMatrix<BaseFloat> delta_bias_trans(output_deriv.NumCols(), 1, kSetZero);
    delta_bias_trans.AddMat(1.0, delta_bias, kTrans);

    to_update->params_.ColRange(params_.NumCols() - 1, 1).AddRows(1.0, delta_bias_trans, idx2);  // TODO(hxu)
  }
}

void AffineImportanceSamplingComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</AffineImportanceSamplingComponent>");
}

void AffineImportanceSamplingComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</AffineImportanceSamplingComponent>");
}

int32 AffineImportanceSamplingComponent::NumParameters() const {
  return (InputDim() ) * ( 1 + OutputDim());
}

void AffineImportanceSamplingComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->CopyRowsFromMat(params_);
}
void AffineImportanceSamplingComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
//  KALDI_ASSERT(params.Dim() == this->NumParameters());
  params_.CopyRowsFromVec(params);
//  bias_params_.Row(0).CopyFromVec(params.Range(InputDim() * OutputDim(),
//                                        OutputDim()));
}

void LinearSigmoidNormalizedComponent::Normalize() {

  if (actual_params_.NumRows() != linear_params_.NumRows() ||
      actual_params_.NumCols() != linear_params_.NumCols()) {
    actual_params_.Resize(linear_params_.NumRows(), linear_params_.NumCols());
//    normalizer_.Resize(linear_params_.NumCols());
  }

  CuMatrix<BaseFloat> ht(linear_params_.NumCols(), linear_params_.NumRows());

  ht.CopyFromMat(linear_params_, kTrans);
  ht.Sigmoid(ht);

  // normalize ht s.t. every row adds to 1
  CuMatrix<BaseFloat> ones(ht.NumCols(), 1);
  CuMatrix<BaseFloat> row_sum(1, ht.NumRows(), kSetZero);
  ones.Set(1.0);
  row_sum.AddMatMat(1.0, ones, kTrans, ht, kTrans, 0.0);
  ht.DivRowsVec(row_sum.Row(0));

  actual_params_.CopyFromMat(ht, kTrans);
//  KALDI_ASSERT(ApproxEqual(actual_params_.Sum(), actual_params_.NumCols()));
}

void LinearSigmoidNormalizedComponent::Scale(BaseFloat scale) {
//  KALDI_ASSERT(is_gradient_);
  linear_params_.Scale(scale);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
//  normalized_ = false;
}

void LinearSigmoidNormalizedComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LinearSigmoidNormalizedComponent *other =
      dynamic_cast<const LinearSigmoidNormalizedComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_); //  TODO(hxu)
//  KALDI_LOG << "sum is " << other->linear_params_.Sum();
//  normalized_ = false;
  Normalize();
}

LinearSigmoidNormalizedComponent::LinearSigmoidNormalizedComponent(const LinearSigmoidNormalizedComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
//    normalizer_(component.normalizer_),
    actual_params_(component.actual_params_) {}
//    normalized_(component.normalized_) { }

LinearSigmoidNormalizedComponent::LinearSigmoidNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
//  temp_linear_params.Row(0).SetZero();

  linear_params_.AddMat(stddev, temp_linear_params);
//  normalized_ = false;
  Normalize();
}

std::string LinearSigmoidNormalizedComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  Matrix<BaseFloat> l(linear_params_);
  PrintParameterStats(stream, "linear-params", l);
  return stream.str();
}

LmComponent* LinearSigmoidNormalizedComponent::Copy() const {
  LinearSigmoidNormalizedComponent *ans = new LinearSigmoidNormalizedComponent(*this);
  return ans;
}

BaseFloat LinearSigmoidNormalizedComponent::DotProduct(const LmComponent &other_in) const {
//  KALDI_ASSERT(is_gradient_); // actually there are more problems here ...
  const LinearSigmoidNormalizedComponent *other =
      dynamic_cast<const LinearSigmoidNormalizedComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LinearSigmoidNormalizedComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
//  linear_params_.Row(0).SetZero();
  linear_params_.Scale(param_stddev);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &linear_params_); // will abort on failure.
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(input_dim, output_dim, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LinearSigmoidNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);
}

void LinearSigmoidNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {
  KALDI_ASSERT(false);
}

void LinearSigmoidNormalizedComponent::Backprop(
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}

void LinearSigmoidNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                bool normalize,
                                                CuMatrixBase<BaseFloat> *out) const {
  out->AddMatMat(1.0, in, kNoTrans, actual_params_, kTrans, 1.0);
  KALDI_ASSERT(ApproxEqual(out->Sum(), out->NumRows()));
  out->ApplyLog();
}

void LinearSigmoidNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                 const vector<int> &indexes,
                                                 vector<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == indexes.size());
  out->resize(indexes.size());

  for (int i = 0; i < indexes.size(); i++) {
    int w = indexes[i];
//    KALDI_LOG << in.Row(i).Sum() << " should be close to 1";
    KALDI_ASSERT(ApproxEqual(in.Row(i).Sum(), 1.0));  // TODO(hxu)
    BaseFloat res = VecVec(in.Row(i), actual_params_.Row(w));
//    KALDI_ASSERT(res >= 0 && res <= 1);
    (*out)[i] = res;
  }
}

void LinearSigmoidNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const vector<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  int k = indexes.size();

  KALDI_ASSERT(input_deriv != NULL);

  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    input_deriv->Row(i).AddVec(output_deriv[i], actual_params_.Row(index));
  }

  LinearSigmoidNormalizedComponent* to_update
             = dynamic_cast<LinearSigmoidNormalizedComponent*>(to_update_0);

  KALDI_ASSERT(to_update != NULL);

  CuMatrix<BaseFloat> aT(actual_params_, kTrans);
  CuMatrix<BaseFloat> dapT(actual_params_, kTrans);
  CuMatrix<BaseFloat> daT(actual_params_, kTrans);
//  aT.SetZero();
  dapT.SetZero();
  daT.SetZero();
  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    daT.ColRange(index, 1).AddVecToCols(output_deriv[i], in_value.Row(i), 1.0);
  }

  // first compute derivative of the normalization
  // daT: div on the normalized matrix
  // dapT: div on the matrix before normalization
  // aT: the "in_value"
  // use the back-prop code of NormalizeOneComponent
  {
    aT.Tanh(aT);
    const CuMatrixBase<BaseFloat> &in_value = aT;
    const CuMatrixBase<BaseFloat> &out_deriv = daT;
    CuMatrixBase<BaseFloat> *in_deriv = &dapT;

    CuMatrix<BaseFloat> ones(in_value.NumCols(), 1);
    ones.Set(1.0);

    CuMatrix<BaseFloat> in_row_sum(1, in_value.NumRows(), kSetZero);
    in_row_sum.AddMatMat(1.0, ones, kTrans, in_value, kTrans, 0.0);

  //  KALDI_ASSERT(ApproxEqual(in_row_sum.Sum(), in_value.Sum()));

    CuMatrix<BaseFloat> t(out_deriv);
    t.MulElements(in_value);
    CuMatrix<BaseFloat> row_sum2(1, in_value.NumRows(), kSetZero);

  //  row_sum2.AddMatMat(1.0, in_value, kNoTrans, ones, kNoTrans, 0.0);
    row_sum2.AddMatMat(1.0, ones, kTrans, t, kTrans, 0.0);

    row_sum2.DivElements(in_row_sum);
    row_sum2.DivElements(in_row_sum);
    row_sum2.Scale(-1);

    in_deriv->AddMatMat(1.0, row_sum2, kTrans, ones, kTrans, 1.0);

  //  KALDI_LOG << "d sum here is " << out_deriv.Sum();
  //  KALDI_LOG << "in sum here is " << in_value.Sum();
  //
  //
  //  KALDI_LOG << "a sum here is " << in_deriv->Sum();

    t.CopyFromMat(out_deriv);
    t.DivRowsVec(in_row_sum.Row(0));
    in_deriv->AddMat(1.0, t);
  }

  // now dapT is the derivative of the tanh'd matrix
  // aT is the tanh/d matrix
  dapT.DiffSigmoid(aT, dapT);

//  dapT.DiffSoftmaxPerRow(aT, daT);
//  KALDI_LOG << aT.Sum() << " and " << daT.Sum() << " and " <<dapT.Sum();

  to_update->linear_params_.AddMat(learning_rate_, dapT, kTrans);  // TODO(hxu)
//  to_update->linear_params_.Row(0).SetZero();
}

void LinearSigmoidNormalizedComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<ActualParams>");
  actual_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LinearSigmoidNormalizedComponent>");
//  Normalize();
}

void LinearSigmoidNormalizedComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<ActualParams>");
  actual_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LinearSigmoidNormalizedComponent>");
}

int32 LinearSigmoidNormalizedComponent::NumParameters() const {
  return InputDim() * OutputDim(); // actually should be (InputDim() - 1 ) * OutputDim()
}

void LinearSigmoidNormalizedComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}

void LinearSigmoidNormalizedComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Normalize() {

  if (actual_params_.NumRows() != linear_params_.NumRows() ||
      actual_params_.NumCols() != linear_params_.NumCols()) {
    actual_params_.Resize(linear_params_.NumRows(), linear_params_.NumCols());
//    normalizer_.Resize(linear_params_.NumCols());
  }

  CuMatrix<BaseFloat> ht(linear_params_.NumCols(), linear_params_.NumRows());

  ht.CopyFromMat(linear_params_, kTrans);
//  ht.AddVecToCols(-1.0, linear_params_.Row(0));

//  linear_params_.CopyFromMat(ht, kTrans);

  ht.ApplySoftMaxPerRow(ht);
//  for (int i = 0; i < ht.NumRows(); i++) {
//    normalizer_(i) = ht.Row(i).ApplySoftMax();
//  }

  actual_params_.CopyFromMat(ht, kTrans);
//  KALDI_ASSERT(ApproxEqual(actual_params_.Sum(), actual_params_.NumCols()));
}

void LinearSoftmaxNormalizedComponent::Scale(BaseFloat scale) {
//  KALDI_ASSERT(is_gradient_);
  linear_params_.Scale(scale);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
//  normalized_ = false;
}

void LinearSoftmaxNormalizedComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LinearSoftmaxNormalizedComponent *other =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_); //  TODO(hxu)
//  KALDI_LOG << "sum is " << other->linear_params_.Sum();
//  normalized_ = false;
  Normalize();
}

LinearSoftmaxNormalizedComponent::LinearSoftmaxNormalizedComponent(const LinearSoftmaxNormalizedComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
//    normalizer_(component.normalizer_),
    actual_params_(component.actual_params_) {}
//    normalized_(component.normalized_) { }

LinearSoftmaxNormalizedComponent::LinearSoftmaxNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  temp_linear_params.Row(0).SetZero();

  linear_params_.AddMat(stddev, temp_linear_params);
//  normalized_ = false;
  Normalize();
}

std::string LinearSoftmaxNormalizedComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  Matrix<BaseFloat> l(linear_params_);
  PrintParameterStats(stream, "linear-params", l);
  return stream.str();
}

LmComponent* LinearSoftmaxNormalizedComponent::Copy() const {
  LinearSoftmaxNormalizedComponent *ans = new LinearSoftmaxNormalizedComponent(*this);
  return ans;
}

BaseFloat LinearSoftmaxNormalizedComponent::DotProduct(const LmComponent &other_in) const {
//  KALDI_ASSERT(is_gradient_); // actually there are more problems here ...
  const LinearSoftmaxNormalizedComponent *other =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LinearSoftmaxNormalizedComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Row(0).SetZero();
  linear_params_.Scale(param_stddev);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &linear_params_); // will abort on failure.
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  std::string unigram_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");

  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);

    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(input_dim, output_dim, param_stddev);

    if (cfl->GetValue("unigram", &unigram_filename)) {
      std::vector<BaseFloat> u;
      ReadUnigram(unigram_filename, &u);
      InitFromUnigram(u);
      PerturbParams(param_stddev);
      Normalize();
    }
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LinearSoftmaxNormalizedComponent::InitFromUnigram(const std::vector<BaseFloat> &unigram) {
  KALDI_ASSERT(unigram.size() == linear_params_.NumRows());
//  for (int i = 0; i < unigram.size(); i++) {
//    unigram[i] = log(unigram[i]);
//  }
  Matrix<BaseFloat> a(unigram.size(), 1, kSetZero);
  for (int i = 0; i < unigram.size(); i++) {
    a(i, 0) = unigram[i];
  }
  a.ApplyLog();
  a.Add(-a(0, 0));
  Matrix<BaseFloat> b(1, linear_params_.NumCols(), kSetZero);
  b.Set(1.0);
  Matrix<BaseFloat> linear(linear_params_.NumRows(), linear_params_.NumCols(), kSetZero);
  linear.AddMatMat(1, a, kNoTrans, b, kNoTrans, 0);
  linear_params_.CopyFromMat(linear);

}

void LinearSoftmaxNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);
}

void LinearSoftmaxNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                bool normalize,
                                                CuMatrixBase<BaseFloat> *out) const {
  out->AddMatMat(1.0, in, kNoTrans, actual_params_, kTrans, 1.0);
  KALDI_ASSERT(ApproxEqual(out->Sum(), out->NumRows()));
  out->ApplyLog();
}

void LinearSoftmaxNormalizedComponent::Backprop(
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}

void LinearSoftmaxNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}


void LinearSoftmaxNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                vector<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == indexes.size());
  out->resize(indexes.size());

  for (int i = 0; i < indexes.size(); i++) {
    int w = indexes[i];
//    KALDI_LOG << in.Row(i).Sum() << " should be close to 1";
    BaseFloat sum = 0.0;
    if (!ApproxEqual(sum = in.Row(i).Sum(), 1.0)) {
      KALDI_LOG << sum << " should be close to 1";
    }
    BaseFloat res = VecVec(in.Row(i), actual_params_.Row(w));
//    KALDI_ASSERT(res >= 0 && res <= 1);
    (*out)[i] = res;
  }
}

void LinearSoftmaxNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const vector<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  int k = indexes.size();

  KALDI_ASSERT(input_deriv != NULL);

  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    input_deriv->Row(i).AddVec(output_deriv[i], actual_params_.Row(index));
  }

  LinearSoftmaxNormalizedComponent* to_update
             = dynamic_cast<LinearSoftmaxNormalizedComponent*>(to_update_0);

  KALDI_ASSERT(to_update != NULL);

  CuMatrix<BaseFloat> aT(actual_params_, kTrans);
  CuMatrix<BaseFloat> dapT(actual_params_, kTrans);
  CuMatrix<BaseFloat> daT(actual_params_, kTrans);
//  aT.SetZero();
  dapT.SetZero();
  daT.SetZero();
  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    daT.ColRange(index, 1).AddVecToCols(output_deriv[i], in_value.Row(i), 1.0);
  }
  dapT.DiffSoftmaxPerRow(aT, daT);
//  KALDI_LOG << aT.Sum() << " and " << daT.Sum() << " and " <<dapT.Sum();
  to_update->linear_params_.AddMat(learning_rate_, dapT, kTrans);  // TODO(hxu)
  to_update->linear_params_.Row(0).SetZero();
}

void LinearSoftmaxNormalizedComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<ActualParams>");
  actual_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LinearSoftmaxNormalizedComponent>");
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<ActualParams>");
  actual_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LinearSoftmaxNormalizedComponent>");
}

int32 LinearSoftmaxNormalizedComponent::NumParameters() const {
  return InputDim() * OutputDim(); // actually should be (InputDim() - 1 ) * OutputDim()
}

void LinearSoftmaxNormalizedComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}

void LinearSoftmaxNormalizedComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  Normalize();
}

void LmLinearComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
}

void LmLinearComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
}

void LmLinearComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LmLinearComponent *other =
      dynamic_cast<const LmLinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
//  bias_params_.AddVec(alpha, other->bias_params_);
}

LmLinearComponent::LmLinearComponent(const LmLinearComponent &component):
    LmInputComponent(component),
    linear_params_(component.linear_params_) {}

LmLinearComponent::LmLinearComponent(const MatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
}

void LmLinearComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
}

void LmLinearComponent::SetParams(//const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
}

void LmLinearComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
}

std::string LmLinearComponent::Info() const {
  std::ostringstream stream;
  stream << LmInputComponent::Info();
  nnet3::PrintParameterStats(stream, "linear-params", linear_params_);
  return stream.str();
}

LmComponent* LmLinearComponent::Copy() const {
  LmLinearComponent *ans = new LmLinearComponent(*this);
  return ans;
}

BaseFloat LmLinearComponent::DotProduct(const LmComponent &other_in) const {
  const LmLinearComponent *other =
      dynamic_cast<const LmLinearComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LmLinearComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {//, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
}

void LmLinearComponent::Init(std::string matrix_filename) {
  // TODO(hxu)
  KALDI_ASSERT(false);
  Matrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
}

void LmLinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
//        bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
//    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim,
         param_stddev);//, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LmLinearComponent::Propagate(const SparseMatrix<BaseFloat> &sp,
                                  CuMatrixBase<BaseFloat> *out) const {
  // out->AddMatMat(1.0, sp, kNoTrans, linear_params_, kTrans, 1.0);

//  CuMatrix<BaseFloat> out2(*out);
//  CuSparseMatrix<BaseFloat> sp_input(sp);
//  CuMatrix<BaseFloat> input(sp.NumRows(), sp.NumCols(), kSetZero);
//  sp_input.CopyToMat(&input);
//  out2.AddMatMat(1.0, input, kNoTrans, linear_params_, kTrans, 1.0);

  cu::ComputeAffineOnSparse(linear_params_, sp, out);

//  BaseFloat r1 = out->Sum();
//  BaseFloat r2 = out2.Sum();
//  KALDI_LOG << r1 << " and " << r2;
//  KALDI_ASSERT(out->ApproxEqual(out2));

}

void LmLinearComponent::UpdateSimple(const SparseMatrix<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  // linear_params_.AddMatMat(learning_rate, out_deriv, kTrans, in_value, kNoTrans, 1.0);
  cu::UpdateSimpleAffineOnSparse(learning_rate_, out_deriv, in_value, &linear_params_);

//std::vector<MatrixIndexT> vis;
//const SparseMatrix<BaseFloat> &sp = in_value;
//
//for (size_t i = 0; i < sp.NumRows(); i++) {
//const SparseVector<BaseFloat> &sv = sp.Row(i);
//int non_zero_index = -1;
//ApproxEqual(sv.Max(&non_zero_index), 1.0);
//vis.push_back(non_zero_index);
//}
//KALDI_ASSERT(vis.size() == sp.NumRows());
//
//// TODO(hxu)
//for (int i = 0; i < vis.size(); i++) {
//MatrixIndexT j = vis[i];
//// i.e. in_value (i, j) = 1
//
//for (int k = 0; k < out_deriv.NumCols(); k++) {
//linear_params_(k, j) += learning_rate_ * out_deriv(i, k);
////      KALDI_LOG << k << ", " << j << " added " << out_deriv(k, i);
//}
//}
}

void LmLinearComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void LmLinearComponent::Backprop(
                               const SparseMatrix<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               LmComponent *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  LmLinearComponent *to_update = dynamic_cast<LmLinearComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                        1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(in_value, out_deriv);  // by child classes.
  }
}

void LmLinearComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
//  ExpectToken(is, binary, "<BiasParams>");
//  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LmLinearComponent>");
}

void LmLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
//  WriteToken(os, binary, "<BiasParams>");
//  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LmLinearComponent>");
}

int32 LmLinearComponent::NumParameters() const {
  return InputDim() * OutputDim();
}
void LmLinearComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}
void LmLinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
}

} // namespace rnnlm
} // namespace kaldi
