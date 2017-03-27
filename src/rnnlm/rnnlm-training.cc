
// Copyright      2016 Hainan Xu

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

#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace rnnlm {

void ComputeSamplingNonlinearity(const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) {
  KALDI_ASSERT(in.NumRows() == out->NumRows() && in.NumCols() == out->NumCols());
  CuMatrix<BaseFloat> tmp(in);
  tmp.ApplyFloor(0);
  // now tmp retains the postive part of in

  out->CopyFromMat(in);
  out->ApplyCeiling(0);
  out->ApplyExp();
  // now out is 1 when positive, exp(x) when negative

  out->AddMat(1.0, tmp);
}

void BackpropSamplingNonlinearity(const CuVectorBase<BaseFloat> &probs_inv,
                                  CuMatrixBase<BaseFloat> *out_value,
                                  CuMatrixBase<BaseFloat> *in_deriv) {
  // out_value is here because we need its value;
  // then we will re-use it as a tmp variable since it's not needed later

  // here we compute the derivative of the
  // f = -\sum f(y_j)/p(y_j) where f is the ComputeSamplingNonlinearity function
  // its derivative is the product of the following 2:
  // 1. \partial f / \partial f(y_j) = -1/p(y_j)
  // 2. \partial f(y_j) / \partial y_j, which is exp(y_j) for negative,
  //                                             1 for positive

  out_value->ApplyCeiling(1.0); // now out_value represents quantities in (2)
  in_deriv->CopyRowsFromVec(probs_inv); // now in_deriv stores (1)
  in_deriv->MulElements(*out_value); // multiply the 2
//  KALDI_LOG << "sum is " << in_deriv->Sum();
}

LmNnetSamplingTrainer::LmNnetSamplingTrainer(
                          const LmNnetTrainerOptions &config,
                          const vector<BaseFloat> &unigram,
                          LmNnet *nnet):
                          config_(config),
                          unigram_(unigram),
                          nnet_(nnet),
                          compiler_(*nnet->GetNnet(), config_.optimize_config),
                          num_minibatches_processed_(0) {
  KALDI_ASSERT(unigram_.size() == nnet_->O()->OutputDim());

  if (config.zero_component_stats)
    nnet->ZeroStats();
  if (config.momentum == 0.0 && config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config.momentum >= 0.0 &&
                 config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    delta_nnet_->SetZero(is_gradient);
    const int32 num_updatable = NumUpdatableComponents(delta_nnet_->Nnet());
    num_max_change_per_component_applied_.resize(num_updatable, 0); 
    num_max_change_per_component_applied_2_.resize(2, 0); 
    num_max_change_global_applied_ = 0;
  }
  if (config_.read_cache != "") {
    bool binary;
    try {
      Input ki(config_.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << config_.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  } 
}

void LmNnetSamplingTrainer::ProcessEgInputs(const NnetExample& eg,
                                            const LmInputComponent& a,
                                            const SparseMatrix<BaseFloat> **old_input,
                                            CuMatrix<BaseFloat> *new_input) {
  for (size_t i = 0; i < eg.io.size(); i++) {
    const NnetIo &io = eg.io[i];

    if (io.name == "input") {
      KALDI_ASSERT(old_input != NULL && new_input != NULL);
      new_input->Resize(io.features.NumRows(),
                        a.OutputDim(),
                        kSetZero);

      *old_input = &io.features.GetSparseMatrix();
      a.Propagate(io.features.GetSparseMatrix(), new_input);
    }
  }
}

void LmNnetSamplingTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetComputationRequest(*nnet_->GetNnet(), eg, need_model_derivative,
                        config_.store_component_stats,
                        &request);

  KALDI_ASSERT(request.inputs.size() == 1);
  request.inputs[0].has_deriv = true;

  const NnetComputation *computation = compiler_.Compile(request);

  if (config_.adversarial_training_scale > 0.0 &&
      num_minibatches_processed_ % config_.adversarial_training_interval == 0) {
    // adversarial training is incompatible with momentum > 0
    KALDI_ASSERT(config_.momentum == 0.0);
    delta_nnet_->FreezeNaturalGradient(true);
    bool is_adversarial_step = true;
    TrainInternal(eg, *computation, is_adversarial_step);
    delta_nnet_->FreezeNaturalGradient(false);
  }

  bool is_adversarial_step = false;
  TrainInternal(eg, *computation, is_adversarial_step);

  num_minibatches_processed_++;

}

void LmNnetSamplingTrainer::TrainInternal(const NnetExample &eg,
                                          const NnetComputation& computation,
                                          bool is_adversarial_step) {
  const SparseMatrix<BaseFloat> *old_input;

  NnetComputer computer(config_.compute_config, computation, *nnet_->GetNnet(),
                        (delta_nnet_ == NULL ? nnet_->GetNnet() :
                               delta_nnet_->GetNnet()));

  ProcessEgInputs(eg, *nnet_->input_projection_, &old_input, &new_input_);

  // give the inputs to the computer object.
  computer.AcceptInput("input", &new_input_);
  computer.Run();

  // in ProcessOutputs() we first do the last Forward propagation
  // and before exiting, do the first step of back-propagation
  this->ProcessOutputs(is_adversarial_step, eg, &computer);
  computer.Run();

  const CuMatrixBase<BaseFloat> &first_deriv = computer.GetOutput("input");
  CuMatrix<BaseFloat> place_holder;
  nnet_->I()->Backprop(*old_input, place_holder,
                       first_deriv, delta_nnet_->input_projection_, NULL);

  UpdateParamsWithMaxChange(is_adversarial_step);
}

void LmNnetSamplingTrainer::UpdateParamsWithMaxChange(bool is_adversarial_step) {
  KALDI_ASSERT(delta_nnet_ != NULL);
  // computes scaling factors for per-component max-change
  const int32 num_updatable = NumUpdatableComponents(delta_nnet_->Nnet());
  Vector<BaseFloat> scale_factors = Vector<BaseFloat>(num_updatable);
  BaseFloat param_delta_squared = 0.0;
  int32 num_max_change_per_component_applied_per_minibatch = 0;
  BaseFloat min_scale = 1.0;
  std::string component_name_with_min_scale;
  BaseFloat max_change_with_min_scale;
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->Nnet().NumComponents(); c++) {
    const Component *comp = delta_nnet_->Nnet().GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      const UpdatableComponent *uc = dynamic_cast<const UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      BaseFloat max_param_change_per_comp = uc->MaxChange();
      KALDI_ASSERT(max_param_change_per_comp >= 0.0);
      BaseFloat dot_prod = uc->DotProduct(*uc);
      if (max_param_change_per_comp != 0.0 &&
          std::sqrt(dot_prod) > max_param_change_per_comp) {
        scale_factors(i) = max_param_change_per_comp / std::sqrt(dot_prod);
        num_max_change_per_component_applied_[i]++;
        num_max_change_per_component_applied_per_minibatch++;
        KALDI_VLOG(2) << "Parameters in " << delta_nnet_->Nnet().GetComponentName(c)
                      << " change too big: " << std::sqrt(dot_prod) << " > "
                      << "max-change=" << max_param_change_per_comp
                      << ", scaling by " << scale_factors(i);
      } else {
        scale_factors(i) = 1.0;
      }
      if  (i == 0 || scale_factors(i) < min_scale) {
        min_scale =  scale_factors(i);
        component_name_with_min_scale = delta_nnet_->Nnet().GetComponentName(c);
        max_change_with_min_scale = max_param_change_per_comp;
      }
      param_delta_squared += std::pow(scale_factors(i),
                                      static_cast<BaseFloat>(2.0)) * dot_prod;
//      KALDI_LOG << "param_delta_squared here is " << param_delta_squared;
      i++;
    }
  }

  BaseFloat scale_f_in = 1.0;
  BaseFloat scale_f_out = 1.0;

  {
    BaseFloat max_change_per = nnet_->input_projection_->MaxChange();
    KALDI_ASSERT(max_change_per >= 0);
    BaseFloat dot_prod = delta_nnet_->input_projection_->DotProduct(*delta_nnet_->input_projection_);

//    KALDI_LOG << "in dot prod is " << dot_prod;

    if (max_change_per != 0.0 &&
        std::sqrt(dot_prod) > max_change_per) {
      scale_f_in = max_change_per / std::sqrt(dot_prod);
      num_max_change_per_component_applied_2_[0]++;
      num_max_change_per_component_applied_per_minibatch++;
      KALDI_VLOG(2) << "Parameters in Input Projection, "
                    << " change too big: " << std::sqrt(dot_prod) << " > "
                    << "max-change=" << max_change_per
                    << ", scaling by " << scale_f_in;
    }

    if (scale_f_in < min_scale) {
      min_scale = scale_f_in;
      component_name_with_min_scale = "rnnlm-input";
      max_change_with_min_scale = max_change_per;
    }
    param_delta_squared += std::pow(scale_f_in, 
                                    static_cast<BaseFloat>(2.0)) * dot_prod;
//    KALDI_LOG << "param_delta_squared in is " << param_delta_squared;
  }

  {
    BaseFloat max_change_per = nnet_->output_projection_->MaxChange();
    KALDI_ASSERT(max_change_per >= 0);
    BaseFloat dot_prod = delta_nnet_->output_projection_->DotProduct(*delta_nnet_->output_projection_);

//    KALDI_LOG << "out dot prod is " << dot_prod;

    if (max_change_per != 0.0 && std::sqrt(dot_prod) > max_change_per) {
      scale_f_out = max_change_per / std::sqrt(dot_prod);
      num_max_change_per_component_applied_2_[1]++;
      num_max_change_per_component_applied_per_minibatch++;
      KALDI_VLOG(2) << "Parameters in Output Projection, "
                    << " change too big: " << std::sqrt(dot_prod) << " > "
                    << "max-change=" << max_change_per
                    << ", scaling by " << scale_f_out;
    }

    if (scale_f_out < min_scale) {
      min_scale = scale_f_out;
      component_name_with_min_scale = "rnnlm-input";
      max_change_with_min_scale = max_change_per;
    }
    param_delta_squared += std::pow(scale_f_out, 
                                    static_cast<BaseFloat>(2.0)) * dot_prod;
//    KALDI_LOG << "scale, dotprod is " << scale_f_out << " " << dot_prod;
//    KALDI_LOG << "param_delta_squared out is " << param_delta_squared;
  }


  KALDI_ASSERT(i == scale_factors.Dim());
  BaseFloat param_delta = std::sqrt(param_delta_squared);
//  KALDI_LOG << "param_delta is " << param_delta;
  // computes the scale for global max-change (with momentum)
  BaseFloat scale = (1.0 - config_.momentum);
  if (config_.max_param_change != 0.0) {
    param_delta *= scale;
    if (param_delta > config_.max_param_change) {
      if (param_delta - param_delta != 0.0) {
        KALDI_WARN << "Infinite parameter change, will not apply.";
        delta_nnet_->SetZero(false);
      } else {
        scale *= config_.max_param_change / param_delta;
        num_max_change_global_applied_++;
      }
    }
  }
  if ((config_.max_param_change != 0.0 &&
      param_delta > config_.max_param_change &&
      param_delta - param_delta == 0.0) || min_scale < 1.0) {
    std::ostringstream ostr;
    if (min_scale < 1.0)
      ostr << "Per-component max-change active on "
           << num_max_change_per_component_applied_per_minibatch
           << " / " << num_updatable + 2 << " Updatable Components."
           << "(smallest factor=" << min_scale << " on "
           << component_name_with_min_scale
           << " with max-change=" << max_change_with_min_scale <<"). "; 
    if (param_delta > config_.max_param_change)
      ostr << "Global max-change factor was "
           << config_.max_param_change / param_delta
           << " with max-change=" << config_.max_param_change << ".";
    KALDI_LOG << ostr.str();
  }
  // applies both of the max-change scalings all at once, component by component
  // and updates parameters

  if (config_.adversarial_training_scale > 0.0) {
      KALDI_ASSERT(config_.momentum == 0.0);
      BaseFloat scale_adversarial =
          (is_adversarial_step ? -config_.adversarial_training_scale :
          (1 + config_.adversarial_training_scale));

      scale_factors.Scale(scale * scale_adversarial);

      AddNnetComponents(delta_nnet_->Nnet(), scale_factors, scale * scale_adversarial,
                        nnet_->nnet_);

      nnet_->input_projection_->Add(scale_f_in * scale_adversarial,
                                    *delta_nnet_->input_projection_);
      nnet_->output_projection_->Add(scale_f_out * scale_adversarial,
                                     *delta_nnet_->output_projection_);

//      ScaleNnet(0.0, delta_nnet_);
      delta_nnet_->Scale(0.0);
    } else {
//      scale_factors.Scale(scale);
//      AddNnetComponents(*delta_nnet_, scale_factors, scale, nnet_);
//      ScaleNnet(config_.momentum, delta_nnet_);
      scale_factors.Scale(scale);
      AddNnetComponents(delta_nnet_->Nnet(), scale_factors, scale, nnet_->nnet_);
      nnet_->input_projection_->Add(scale_f_in, *delta_nnet_->input_projection_);
      nnet_->output_projection_->Add(scale_f_out, *delta_nnet_->output_projection_);
    //  ScaleNnet(config_.momentum, delta_nnet_);
      delta_nnet_->Scale(config_.momentum);
    }
  }
//}

void LmNnetSamplingTrainer::ProcessOutputs(bool is_adversarial_step,
                                           const NnetExample &eg,
                                           NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(), end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNnet()->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->GetNnet()->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNnet()->GetNode(node_index).u.objective_type;
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;

      if (dynamic_cast<const AffineImportanceSamplingComponent*>(nnet_->O()) != NULL) {
        ComputeObjectiveFunctionSample(config_.sample_size, unigram_, io.features,
                                       obj_type, io.name,
                                       supply_deriv, computer,
                                       &tot_weight, &tot_objf,
                                       *nnet_->O(),
                                       &old_output_, delta_nnet_);
      } else if (dynamic_cast<const LinearSoftmaxNormalizedComponent*>(nnet_->O()) != NULL) {
        ComputeObjectiveFunctionNormalized(io.features, obj_type, io.name,
                                           supply_deriv, computer,
                                           &tot_weight, &tot_objf,
                                           *nnet_->O(),
                                           &old_output_, delta_nnet_);
      } else if (dynamic_cast<const LinearSigmoidNormalizedComponent*>(nnet_->O()) != NULL) {
        ComputeObjectiveFunctionNormalized(io.features, obj_type, io.name,
                                           supply_deriv, computer,
                                           &tot_weight, &tot_objf,
                                           *nnet_->O(),
                                           &old_output_, delta_nnet_);
      } else {
        KALDI_ASSERT(false);
      }

      objf_info_[io.name].UpdateStats(io.name, config_.print_interval,
                                      num_minibatches_processed_,
                                      tot_weight, tot_objf);
    }
  }
}

bool LmNnetSamplingTrainer::PrintTotalStats() const {
  unordered_map<std::string, LmObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const LmObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}

void LmObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_objf,
    BaseFloat this_minibatch_tot_aux_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase >= current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, minibatches_per_phase);
    current_phase = phase;
    tot_weight_this_phase = 0.0;
    tot_objf_this_phase = 0.0;
    tot_aux_objf_this_phase = 0.0;
  }
  tot_weight_this_phase += this_minibatch_weight;
  tot_objf_this_phase += this_minibatch_tot_objf;
  tot_aux_objf_this_phase += this_minibatch_tot_aux_objf;
  tot_weight += this_minibatch_weight;
  tot_objf += this_minibatch_tot_objf;
  tot_aux_objf += this_minibatch_tot_aux_objf;
}

void LmObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    int32 minibatches_per_phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = start_minibatch + minibatches_per_phase - 1;

  if (tot_aux_objf_this_phase == 0.0) {
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << (tot_objf_this_phase / tot_weight_this_phase) << " over "
              << tot_weight_this_phase << " frames.";
  } else {
    BaseFloat objf = (tot_objf_this_phase / tot_weight_this_phase),
        aux_objf = (tot_aux_objf_this_phase / tot_weight_this_phase),
        sum_objf = objf + aux_objf;
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << objf << " + " << aux_objf << " = " << sum_objf
              << " over " << tot_weight_this_phase << " frames.";
  }
}

bool LmObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
  BaseFloat objf = (tot_objf / tot_weight),
        aux_objf = (tot_aux_objf / tot_weight),
        sum_objf = objf + aux_objf;
  if (tot_aux_objf == 0.0) {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << (tot_objf / tot_weight) << " over " << tot_weight << " frames.";
  } else {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << objf << " + " << aux_objf << " = " << sum_objf        
              << " over " << tot_weight << " frames.";
  }
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "log-prob-per-frame="
            << objf;
  return (tot_weight != 0.0);
}

LmNnetSamplingTrainer::~LmNnetSamplingTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  } 
  delete delta_nnet_;
}

void LmNnetSamplingTrainer::ComputeObjectiveFunctionNormalized(
                              const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf,
                              const LmOutputComponent &output_projection,
                              const CuMatrixBase<BaseFloat> **old_output,
                              LmNnet *nnet) {
  *old_output = &computer->GetOutput(output_name);
  int k = supervision.NumRows();

  KALDI_ASSERT(supervision.Type() == kSparseMatrix);
  const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();

  std::vector<int> indexes;

  SparseMatrixToVector(post, &indexes);

  vector<BaseFloat> out(indexes.size());

  {
    const LinearSoftmaxNormalizedComponent* output_project_norm =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&output_projection);
    if (output_project_norm != NULL) {
      output_project_norm->Propagate(**old_output, indexes, &out);
    }
  }
  {
    const LinearSigmoidNormalizedComponent* output_project_norm =
      dynamic_cast<const LinearSigmoidNormalizedComponent*>(&output_projection);
    if (output_project_norm != NULL) {
      output_project_norm->Propagate(**old_output, indexes, &out);
    }
  }

  *tot_weight = post.Sum();
  *tot_objf = 0;
  
  for (int i = 0; i < k; i++) {
    *tot_objf += log(out[i]);
  }

  if (supply_deriv && nnet != NULL) {
  // the derivative on the real output
    vector<BaseFloat> output_deriv(k);

    for (int i = 0; i < k; i++) {
      output_deriv[i] = 1.0 / out[i];
    }

    // the derivative after the affine layer (before the nonlin)

    // the derivative of the 'nnet3' part
    CuMatrix<BaseFloat> input_deriv((*old_output)->NumRows(),
        (*old_output)->NumCols(),
        kSetZero);

    CuMatrix<BaseFloat> place_holder;

    {
      const LinearSoftmaxNormalizedComponent* output_project_norm =
        dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&output_projection);
      if (output_project_norm != NULL) {
        output_project_norm->Backprop(indexes, **old_output, place_holder,
            output_deriv, nnet->output_projection_,
            &input_deriv);
      }
    }
    {
      const LinearSigmoidNormalizedComponent* output_project_norm =
        dynamic_cast<const LinearSigmoidNormalizedComponent*>(&output_projection);
      if (output_project_norm != NULL) {
        output_project_norm->Backprop(indexes, **old_output, place_holder,
            output_deriv, nnet->output_projection_,
            &input_deriv);
      }
    }

    computer->AcceptInput(output_name, &input_deriv);
  }
}

void LmNnetSamplingTrainer::ComputeObjectiveFunctionSample(
                              int num_samples,
                              const vector<BaseFloat> &unigram,
                              const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf,
                              const LmOutputComponent &output_projection,
                              const CuMatrixBase<BaseFloat> **old_output,
                              LmNnet *nnet) {
  *old_output = &computer->GetOutput(output_name);
  int k = supervision.NumRows();
  if (num_samples == -1) {
    num_samples = unigram.size();
  }

  KALDI_ASSERT(num_samples > k && num_samples <= unigram.size());

  KALDI_ASSERT(supervision.Type() == kSparseMatrix);
  const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();

  vector<BaseFloat> selection_probs = unigram;

  std::vector<int> outputs;  // outputs[i] is the correct work for row i
  std::set<int> outputs_set;

  SparseMatrixToVector(post, &outputs);

  vector<std::pair<int, BaseFloat> > samples(num_samples);
  vector<BaseFloat> selected_probs(num_samples);

  if (num_samples != unigram.size()) {
    for (int i = 0; i < outputs.size(); i++) {
      outputs_set.insert(outputs[i]);
    }
    NormalizeVec(num_samples, outputs_set, &selection_probs);
    vector<BaseFloat> u(selection_probs.size());
    for (int i = 0; i < u.size(); i++) {
//      u[i].first = i;
//      u[i].second = selection_probs[i];
      u[i] = selection_probs[i];
    }
    SampleWithoutReplacement(u, num_samples, &samples);
    for (int i = 0; i < samples.size(); i++) {
      // use min with 1.0 because we use prob=10 for "must sampled" words
      // in the vector to avoid numerical issues
      selected_probs[i] = std::min(BaseFloat(1.0), selection_probs[samples[i].first]);
    }
  } else {
    for (int i = 0; i < num_samples; i++) {
      samples[i].first = i;
      samples[i].second = 1.0;
    }
  }

  CuMatrix<BaseFloat> out((*old_output)->NumRows(), samples.size(), kSetZero);
//  output_projection.Propagate(**old_output, samples, &out);
//
//  CuMatrix<BaseFloat> f_out(out.NumRows(), out.NumCols());
//  ComputeSamplingNonlinearity(out, &f_out);
//
//  *tot_weight = post.Sum();
//  vector<int32> correct_indexes(out.NumRows(), -1);
//
//  if (num_samples == unigram.size()) {
//    for (int j = 0; j < outputs.size(); j++) {
//      correct_indexes[j] = outputs[j];
//    }
//  } else {
//    // TODO(hxu) not tested it yet
//    unordered_map<int32, int32> word2pos;
//    for (int i = 0; i < samples.size(); i++) {
//      word2pos[samples[i]] = i;
//    }
//    for (int i = 0; i < outputs.size(); i++) {
//      correct_indexes[i] = word2pos[outputs[i]];
//    }
//  }
//
//  SparseMatrix<BaseFloat> supervision_cpu;
//  VectorToSparseMatrix(correct_indexes, out.NumCols(), &supervision_cpu);
//  CuSparseMatrix<BaseFloat> supervision_gpu(supervision_cpu);
//  *tot_objf = TraceMatSmat(out, supervision_gpu, kTrans); // first part of the objf
//  // (the objf regarding the positive reward for getting correct labels)
//  // now for each row the objf is y_i where i is the correct label
//  // we need to compute y_i - (\sum_j f(y_j)) + 1
//  // or in the sampling case, y_i - (\sum_j f(y_j)/prob(j))
//
//  // the adjusted output by multiplying by -1/prob(sampling)
//  CuMatrix<BaseFloat> f_out_div_probs(out.NumRows(), out.NumCols());
//  CuVector<BaseFloat> selection_probs_inv(out.NumCols());
//
//  // first fill in the -1/probs
//  if (num_samples != unigram.size()) {
//    Vector<BaseFloat> v(out.NumCols(), kSetZero);
//    for (int i = 0; i < out.NumCols(); i++) {
//      v(i) = -1.0 / selected_probs[i];
//    }
//    selection_probs_inv.CopyFromVec(v);
//    f_out_div_probs.CopyRowsFromVec(selection_probs_inv);
//  } else {
//    f_out_div_probs.Set(-1.0);
//    selection_probs_inv.Set(-1.0);
//  }
//  // now both stores the probs only
//
//  // now multiply by f(y_i)
//  f_out_div_probs.MulElements(f_out);
//  // now each element is -f(y_i)/selection-prob
//
//  // need to add 1 per row
//  BaseFloat neg_term = f_out_div_probs.Sum() + f_out_div_probs.NumRows();
//  *tot_objf += neg_term;
//
//  if (supply_deriv && nnet != NULL) {
//    CuMatrix<BaseFloat> f_out_div_probs_deriv(out.NumRows(), out.NumCols());
//    BackpropSamplingNonlinearity(selection_probs_inv, &f_out, &f_out_div_probs_deriv);
//
//    CuMatrix<BaseFloat> derivatives;
//    derivatives.Swap(&f_out); // re-use the mem so that no need to malloc again
//    supervision_gpu.CopyToMat(&derivatives); // setting 1 for the correct labels
//    // derivative now has 1 for correct labels
//
//    derivatives.AddMat(1.0, f_out_div_probs_deriv);
//
//    CuMatrix<BaseFloat> input_deriv((*old_output)->NumRows(),
//                                    (*old_output)->NumCols(),
//                                    kSetZero);
//
//    output_projection.Backprop(samples, **old_output, out,
//                               derivatives, nnet->output_projection_,
//                               &input_deriv);
//
////    BaseFloat t = TraceMatMat(nnet->output_projection_->params_, nnet->output_projection_->params_, kTrans);
//
//    computer->AcceptInput(output_name, &input_deriv);
//  }
}

void LmNnetSamplingTrainer::ComputeObjectiveFunctionExact(
                              bool normalize,
                              const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf,
                              const LmOutputComponent &output_projection,
                              CuMatrix<BaseFloat> *new_output,
                              LmNnet *nnet) {
//  *new_output = computer->GetOutput(output_name);
  const CuMatrixBase<BaseFloat> &old_output = computer->GetOutput(output_name);
//  computer->GetOutputDestructive(output_name, &old_output);

  KALDI_ASSERT(supervision.Type() == kSparseMatrix);
  const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
  CuSparseMatrix<BaseFloat> cu_post(post);

  {
    const AffineImportanceSamplingComponent* output_project =
      dynamic_cast<const AffineImportanceSamplingComponent*>(&output_projection);
    if (output_project != NULL) {
      output_project->Propagate(old_output, normalize, new_output);
    }
  }
  {
    const LinearSoftmaxNormalizedComponent* output_project =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&output_projection);
    if (output_project != NULL) {
      output_project->Propagate(old_output, normalize, new_output);
    }
  }

  *tot_weight = post.Sum();
  *tot_objf = TraceMatSmat(*new_output, cu_post, kTrans);

  if (supply_deriv && nnet != NULL) {

    CuMatrix<BaseFloat> output_deriv(new_output->NumRows(), new_output->NumCols(),
                                     kUndefined);
    cu_post.CopyToMat(&output_deriv);

    CuMatrix<BaseFloat> input_deriv(new_output->NumRows(),
                                    output_projection.InputDim(),
                                    kSetZero);

    output_projection.Backprop(old_output, *new_output,
                               output_deriv, nnet->output_projection_,
                               &input_deriv);

    computer->AcceptInput(output_name, &input_deriv);
  }

}

} // namespace nnet3
} // namespace kaldi
