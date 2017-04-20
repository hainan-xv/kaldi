// nnet3/nnet-component-itf.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen
//                2015  Xiaohui Zhang

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

#ifndef KALDI_RNNLM_RNNLM_COMPONENT_ITF_H_
#define KALDI_RNNLM_RNNLM_COMPONENT_ITF_H_

#include "nnet3/nnet-common.h"
#include "rnnlm/nnet-parse.h"
#include "nnet3/nnet-parse.h"
#include "base/kaldi-error.h"
#include "thread/kaldi-mutex.h"

//#include "nnet3/nnet-component-itf.h"
//#include "rnnlm/rnnlm-component-itf.h"
//#include "nnet3/nnet-computation-graph.h"
#include <iostream>

namespace kaldi {

namespace nnet3 {
  class IndexSet;
}

namespace rnnlm {


using std::vector;
using nnet3::MiscComputationInfo;
using nnet3::Index;
using nnet3::ConfigLine;
using nnet3::SummarizeVector;
using nnet3::ExpectOneOrTwoTokens;

enum ComponentProperties {
  kSimpleComponent = 0x001,  // true if number of rows of input equals number of rows
                             // of output and this component doesn't care about the indexes
                             // (i.e. it maps each row of input to each row of output without
                             // regard to the index values).  Will normally be true.
  kUpdatableComponent = 0x002,  // true if the component has parameters that can
                                // be updated.  Components that return this flag
                                // must be dynamic_castable to type
                                // UpdatableComponent (but components of type
                                // UpdatableComponent do not have to return this
                                // flag, e.g.  if this instance is not really
                                // updatable).
  kLinearInInput = 0x004,    // true if the component's output is always a
                             // linear function of its input, i.e. alpha times
                             // input gives you alpha times output.
  kLinearInParameters = 0x008, // true if an updatable component's output is always a
                               // linear function of its parameters, i.e. alpha times
                               // parameters gives you alpha times output.  This is true
                               // for all updatable components we envisage.
  kPropagateInPlace = 0x010,  // true if we can do the propagate operation in-place
                              // (input and output matrices are the same).
                              // Note: if doing backprop, you'd also need to check
                              // that the kBackpropNeedsInput property is not true.
  kPropagateAdds = 0x020,  // true if the Propagate function adds to, rather
                           // than setting, its output.  The Component chooses
                           // whether to add or set, and the calling code has to
                           // accommodate it.
  kReordersIndexes = 0x040,  // true if the ReorderIndexes function might reorder
                             // the indexes (otherwise we can skip calling it).
                             // Must not be set for simple components.
  kBackpropAdds = 0x080,   // true if the Backprop function adds to, rather than
                           // setting, the "in_deriv" output.  The Component
                           // chooses whether to add or set, and the calling
                           // code has to accommodate it.  Note: in the case of
                           // in-place backprop, this flag has no effect.
  kBackpropNeedsInput = 0x100,  // true if backprop operation needs access to
                                // forward-pass input.
  kBackpropNeedsOutput = 0x200,  // true if backprop operation needs access to
                                 // forward-pass output (e.g. true for Sigmoid).
  kBackpropInPlace = 0x400,   // true if we can do the backprop operation in-place
                             // (input and output matrices may be the same).
  kStoresStats = 0x800,      // true if the StoreStats operation stores
                             // statistics e.g. on average node activations and
                             // derivatives of the nonlinearity, (as it does for
                             // Tanh, Sigmoid, ReLU and Softmax).
  kInputContiguous = 0x1000,  // true if the component requires its input data (and
                              // input derivatives) to have Stride()== NumCols().
  kOutputContiguous = 0x2000  // true if the component requires its input data (and
                              // output derivatives) to have Stride()== NumCols().
};

class LmInputComponent {
 public:
  LmInputComponent(const LmInputComponent &other) {
    learning_rate_ = other.learning_rate_;
    learning_rate_factor_ = other.learning_rate_factor_;
    is_gradient_ = other.is_gradient_;
    max_change_ = other.max_change_;
  }

  /// \brief Sets parameters to zero, and if treat_as_gradient is true,
  ///  sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  ///  learning_rate_factor_.
  virtual void SetZero(bool treat_as_gradient) = 0;

  virtual int32 InputDim() const = 0;
  virtual int32 OutputDim() const = 0;
  BaseFloat LearningRate() const { return learning_rate_; }
  BaseFloat MaxChange() const { return max_change_; }

  virtual LmInputComponent* Copy() const = 0;

  LmInputComponent() {}

  virtual void Add(BaseFloat alpha, const LmInputComponent &other) = 0;
  virtual void Scale(BaseFloat scale) = 0;
  virtual void FreezeNaturalGradient(bool freeze) {}
  virtual void ZeroStats() { }

  virtual void SetUnderlyingLearningRate(BaseFloat lrate) {
    learning_rate_ = lrate * learning_rate_factor_;
  }

  virtual BaseFloat DotProduct(const LmInputComponent &other) const = 0;

  virtual int32 NumParameters() const { KALDI_ASSERT(0); return 0; }

  static LmInputComponent* ReadNew(std::istream &is, bool binary);
  virtual void Read(std::istream &is, bool binary) = 0;
  virtual void Write(std::ostream &os, bool binary) const = 0;

  static LmInputComponent *NewComponentOfType(const std::string &type);

  virtual void SetActualLearningRate(BaseFloat lrate) { learning_rate_ = lrate; }

  virtual void InitFromConfig(ConfigLine *cfl) = 0;

  virtual ~LmInputComponent() { }

  virtual void Propagate(const SparseMatrix<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const = 0;

  virtual void Backprop(
                        const SparseMatrix<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmInputComponent *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv = NULL) const = 0;

  virtual std::string Info() const;

  virtual std::string Type() const = 0;

 protected:
  // to be called from child classes, extracts any learning rate information
  // from the config line and sets them appropriately.
  void InitLearningRatesFromConfig(ConfigLine *cfl);

  // To be used in child-class Read() functions, this function reads the opening
  // tag <ThisComponentType> and the learning-rate factor and the learning-rate.
  void ReadUpdatableCommon(std::istream &is, bool binary);

  // To be used in child-class Write() functions, writes the opening
  // <ThisComponentType> tag and the learning-rate factor (if not 1.0) and the
  // learning rate;
  void WriteUpdatableCommon(std::ostream &is, bool binary) const;

  BaseFloat learning_rate_;
  BaseFloat learning_rate_factor_;
  bool is_gradient_;
  BaseFloat max_change_;

 private:
  const LmInputComponent &operator = (const LmInputComponent &other); // Disallow.
};


class LmOutputComponent {
 public:
  LmOutputComponent(const LmOutputComponent &other) {
    learning_rate_ = other.learning_rate_;
    learning_rate_factor_ = other.learning_rate_factor_;
    is_gradient_ = other.is_gradient_;
    max_change_ = other.max_change_;
  }

  /// \brief Sets parameters to zero, and if treat_as_gradient is true,
  ///  sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  ///  learning_rate_factor_.
  virtual void SetZero(bool treat_as_gradient) = 0;

  virtual int32 InputDim() const = 0;
  virtual int32 OutputDim() const = 0;

  virtual LmOutputComponent* Copy() const = 0;

  BaseFloat LearningRate() const { return learning_rate_; }
  BaseFloat MaxChange() const { return max_change_; }

  LmOutputComponent() {}

  virtual void Add(BaseFloat alpha, const LmOutputComponent &other) = 0;
  virtual void Scale(BaseFloat scale) = 0;
  virtual void FreezeNaturalGradient(bool freeze) {}
  virtual void ZeroStats() { }

  virtual void SetUnderlyingLearningRate(BaseFloat lrate) {
    learning_rate_ = lrate * learning_rate_factor_;
  }

  virtual BaseFloat DotProduct(const LmOutputComponent &other) const = 0;

  virtual int32 NumParameters() const { KALDI_ASSERT(0); return 0; }

  static LmOutputComponent* ReadNew(std::istream &is, bool binary);
  virtual void Read(std::istream &is, bool binary) = 0;
  virtual void Write(std::ostream &os, bool binary) const = 0;

  static LmOutputComponent *NewComponentOfType(const std::string &type);

  virtual void SetActualLearningRate(BaseFloat lrate) { learning_rate_ = lrate; }

  virtual void InitFromConfig(ConfigLine *cfl) = 0;

  virtual ~LmOutputComponent() { }

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                 const vector<int> &indexes, // objf is computed on the chosen indexes
                 CuMatrixBase<BaseFloat> *out) const = 0;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                 CuMatrixBase<BaseFloat> *out) const = 0;
  
  virtual void Backprop(
             const CuMatrixBase<BaseFloat> &in_value,
             const CuMatrixBase<BaseFloat> &, // out_value
             const CuMatrixBase<BaseFloat> &out_deriv,
             LmOutputComponent *to_update_in,
             CuMatrixBase<BaseFloat> *in_deriv) const = 0;

  virtual void Backprop(
             const vector<int> &indexes,
             const CuMatrixBase<BaseFloat> &in_value,
             const CuMatrixBase<BaseFloat> &, // out_value
             const CuMatrixBase<BaseFloat> &out_deriv,
             LmOutputComponent *to_update_in,
             CuMatrixBase<BaseFloat> *in_deriv) const = 0;

  virtual std::string Info() const;

  virtual std::string Type() const = 0;

 protected:
  // to be called from child classes, extracts any learning rate information
  // from the config line and sets them appropriately.
  void InitLearningRatesFromConfig(ConfigLine *cfl);

  // To be used in child-class Read() functions, this function reads the opening
  // tag <ThisComponentType> and the learning-rate factor and the learning-rate.
  void ReadUpdatableCommon(std::istream &is, bool binary);

  // To be used in child-class Write() functions, writes the opening
  // <ThisComponentType> tag and the learning-rate factor (if not 1.0) and the
  // learning rate;
  void WriteUpdatableCommon(std::ostream &is, bool binary) const;

  BaseFloat learning_rate_;
  BaseFloat learning_rate_factor_;
  bool is_gradient_;
  BaseFloat max_change_;
 private:
  const LmOutputComponent &operator = (const LmOutputComponent &other); // Disallow.
};


} // namespace nnet3
} // namespace kaldi


#endif
