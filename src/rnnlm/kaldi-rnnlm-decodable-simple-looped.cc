// rnnlm/kaldi-rnnlm-decodable-simple-looped.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)
//                2017  Yiming Wang

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

#include "rnnlm/kaldi-rnnlm-decodable-simple-looped.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compile-looped.h"

namespace kaldi {
namespace nnet3 {


DecodableRnnlmSimpleLoopedInfo::DecodableRnnlmSimpleLoopedInfo(
    const DecodableRnnlmSimpleLoopedComputationOptions &opts,
    const rnnlm::LmNnet &lm_nnet):
    opts(opts), lm_nnet(lm_nnet) {
  Init(opts, lm_nnet);
}

void DecodableRnnlmSimpleLoopedInfo::Init(
    const DecodableRnnlmSimpleLoopedComputationOptions &opts,
    const rnnlm::LmNnet &lm_nnet) {
  opts.Check();
  KALDI_ASSERT(IsSimpleNnet(lm_nnet.Nnet()));
  int32 left_context, right_context;
  ComputeSimpleNnetContext(lm_nnet.Nnet(), &left_context, &right_context);
  frames_left_context = opts.extra_left_context_initial + left_context;
  frames_right_context = right_context;
  int32 frame_subsampling_factor = 1;
  frames_per_chunk = GetChunkSize(lm_nnet.Nnet(), frame_subsampling_factor,
                                  opts.frames_per_chunk);
  KALDI_ASSERT(frames_per_chunk == opts.frames_per_chunk);
  nnet_output_dim = lm_nnet.Nnet().OutputDim("output");
  KALDI_ASSERT(nnet_output_dim > 0);

  int32 ivector_period = 0;
  int32 extra_right_context = 0;
  int32 num_sequences = 1;  // we're processing one utterance at a time.
  CreateLoopedComputationRequestSimple(lm_nnet.Nnet(), frames_per_chunk,
                                       frame_subsampling_factor,
                                       ivector_period,
                                       opts.extra_left_context_initial,
                                       extra_right_context,
                                       num_sequences,
                                       &request1, &request2, &request3);

  CompileLooped(lm_nnet.Nnet(), opts.optimize_config, request1, request2,
                request3, &computation);
  computation.ComputeCudaIndexes();
  if (GetVerboseLevel() >= 3) {
    KALDI_VLOG(3) << "Computation is:";
    computation.Print(std::cerr, lm_nnet.Nnet());
  }
}

DecodableRnnlmSimpleLooped::DecodableRnnlmSimpleLooped(
    const DecodableRnnlmSimpleLoopedInfo &info) :
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.lm_nnet.Nnet(), NULL),  // NULL is 'nnet_to_update'
    // since everytime we provide one chunk to the decodable object, the size of
    // feats_ == frames_per_chunk
    feats_(info_.frames_per_chunk,
           info_.lm_nnet.InputLayer()->InputDim()),
    current_log_post_offset_(-1) {
  num_frames_ = feats_.NumRows();
}

void DecodableRnnlmSimpleLooped::TakeFeatures(
    const std::vector<int32> &word_indexes) {
  KALDI_ASSERT(word_indexes.size() == num_frames_);
  std::vector<std::vector<std::pair<MatrixIndexT, BaseFloat> > >
      pairs(word_indexes.size());
  for (int32 i = 0; i < word_indexes.size(); i++) {
    std::pair<MatrixIndexT, BaseFloat> one_hot_index(word_indexes[i], 1.0);
    std::vector<std::pair<MatrixIndexT, BaseFloat> > row(1, one_hot_index);
    pairs[i] = row;
  }
  SparseMatrix<BaseFloat> feats_temp(feats_.NumCols(), pairs);
  feats_.Swap(&feats_temp);
  // resets offset so that AdvanceChunk would be called in GetOutput() and
  // GetNnetOutputForFrame()
  current_log_post_offset_ = -1;
}

void DecodableRnnlmSimpleLooped::GetNnetOutputForFrame(
    int32 frame, VectorBase<BaseFloat> *output) {
  KALDI_ASSERT(frame >= 0 && frame < feats_.NumRows());
  if (frame >= current_log_post_offset_ + current_nnet_output_.NumRows())
    AdvanceChunk();
  output->CopyFromVec(current_nnet_output_.Row(frame -
                                               current_log_post_offset_));
}

BaseFloat DecodableRnnlmSimpleLooped::GetOutput(int32 frame, int32 word_index) {
  KALDI_ASSERT(frame >= 0 && frame < feats_.NumRows());
  if (frame >= current_log_post_offset_ + current_nnet_output_.NumRows())
    AdvanceChunk();

  const rnnlm::LmOutputComponent* output_layer = info_.lm_nnet.OutputLayer();
  CuMatrix<BaseFloat> current_nnet_output_gpu;
  current_nnet_output_gpu.Swap(&current_nnet_output_);
  const CuSubVector<BaseFloat> hidden(current_nnet_output_gpu,
                                      frame - current_log_post_offset_);
  BaseFloat log_prob =
      output_layer->ComputeLogprobOfWordGivenHistory(hidden, word_index);
  // swap the pointer back so that this function can be called multiple times
  // with the same returned value before taking next new feats
  current_nnet_output_.Swap(&current_nnet_output_gpu);
  return log_prob;
}

void DecodableRnnlmSimpleLooped::AdvanceChunk() {
  int32 begin_input_frame, end_input_frame;
  begin_input_frame = -info_.frames_left_context;
  // note: end is last plus one.
  end_input_frame = info_.frames_per_chunk + info_.frames_right_context;

  SparseMatrix<BaseFloat> feats_chunk(end_input_frame - begin_input_frame,
                                      feats_.NumCols());
  int32 num_features = feats_.NumRows();
  for (int32 r = begin_input_frame; r < end_input_frame; r++) {
    int32 input_frame = r;
    if (input_frame < 0) input_frame = 0;
    if (input_frame >= num_features) input_frame = num_features - 1;
    feats_chunk.SetRow(r - begin_input_frame, feats_.Row(input_frame));
  }

  const rnnlm::LmInputComponent* input_layer = info_.lm_nnet.InputLayer();
  CuMatrix<BaseFloat> new_input(feats_chunk.NumRows(), input_layer->OutputDim());
  input_layer->Propagate(feats_chunk, &new_input);

  computer_.AcceptInput("input", &new_input);

  computer_.Run();

  {
    // Note: it's possible in theory that if you had weird recurrence that went
    // directly from the output, the call to GetOutputDestructive() would cause
    // a crash on the next chunk.  If that happens, GetOutput() should be used
    // instead of GetOutputDestructive().  But we don't anticipate this will
    // happen in practice.
    CuMatrix<BaseFloat> output;
    computer_.GetOutputDestructive("output", &output);

    current_nnet_output_.Resize(0, 0);
    current_nnet_output_.Swap(&output);
  }
  KALDI_ASSERT(current_nnet_output_.NumRows() == info_.frames_per_chunk &&
               current_nnet_output_.NumCols() == info_.nnet_output_dim);

  current_log_post_offset_ = 0;
}


} // namespace nnet3
} // namespace kaldi
