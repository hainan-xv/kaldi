# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# this script trains a vanilla RNNLM with TensorFlow. 
# to call the script, do
# python steps/tfrnnlm/vanilla_rnnlm.py --data-path=$datadir \
#        --save-path=$savepath --vocab-path=$rnn.wordlist [--hidden-size=$size]
#
# One example recipe is at egs/ami/s5/local/tfrnnlm/run_vanilla_rnnlm.sh

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0,"/home/hxu/.local/lib/python2.7/site-packages/")

import inspect
import time

import numpy as np
import tensorflow as tf

#import reader
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").split()

def _build_vocab(filename):
  words = _read_words(filename)
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model-path", None,
                    "Model output directory.")
flags.DEFINE_string("text-path", None,
                    "text file to score.")
flags.DEFINE_string("wordmap-path", None,
                    "map file from word to word-id.")
flags.DEFINE_integer("batch-size", 64,
                    "batch size for RNN computation.")

FLAGS = flags.FLAGS

def main(_):
  if not FLAGS.model_path:
    raise ValueError("Must set --model-path to the trained RNNLM model")

  wordmap = _build_vocab(FLAGS.wordmap_path)

  oos_id = wordmap["<oos>"]

  batch_size = FLAGS.batch_size

  with tf.Session() as session:
    # restore the model
    saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
    saver.restore(session, FLAGS.model_path)

#    for i in range(0, 10):
##      for j in range(0, 10000):
#      x = np.array([[0, i], [0, i], [0, i+1]], dtype=np.int32)
#      y = np.array([[i, 0], [i, 0], [i+1, 0]], dtype=np.int32)
#      lengths = np.array([[1, 1], [1, 1], [1, 1]])
#
#      result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/bos_sentences:0": x, "Train/Model/sentences_eos:0": y, "Train/Model/test_sentence_lengths:0": lengths})
#      print (result)
#    return

    sentences = []
    sentences_lengths = []
    max_length = 0
    with open(FLAGS.text_path, "r") as f:
      for line in f:
        word_ids = []
        ones = [1]
        for word in line.split():
          ones.append(1)
          if word in wordmap:
            word_ids.append(wordmap[word])
          else:
            word_ids.append(oos_id)
        
        if len(word_ids) > max_length:
          max_length = len(word_ids)
        sentences.append(word_ids)
        sentences_lengths.append(ones)

#        print (word_ids)
        if len(sentences) == batch_size:
          for i in range(0, len(sentences)):
            for n in range(len(sentences[i]), max_length):
              sentences[i].append(0) # pad zeros
              sentences_lengths[i].append(0) # pad zeros
#          print (sentences)
#          print (sentences_lengths)
          raw_sentence = np.asarray(sentences, dtype=np.int32)
#          print ("tensor is: ", raw_sentence)
          bos_sentence = np.concatenate((np.zeros((len(sentences), 1), dtype=np.int32), sentences), axis=1)
#          bos_sentence = np.delete(bos_sentence, max_length, 1)
#          print ("bos tensor is: ", bos_sentence)
          sentence_eos = np.concatenate((sentences, np.zeros((len(sentences), 1), dtype=np.int32)), axis=1)
#          sentence_eos = np.delete(sentence_eos, 0, 1)
#          print ("eos tensor is: ", sentence_eos)
          sentence_lengths = np.asarray(sentences_lengths, dtype=np.int32)
#          print ("length tensor is: ", sentences_lengths)
          result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/bos_sentences:0": bos_sentence, "Train/Model/sentences_eos:0": sentence_eos, "Train/Model/test_sentence_lengths:0": sentence_lengths})
          if result.shape[0] != batch_size:
            print ("wrong dimensions!")
            exit(1)
          print (result)

          sentences = []
          sentences_lengths = []
          max_length = 0

      if len(sentences) != 0:
        for i in range(0, len(sentences)):
          for n in range(len(sentences[i]), max_length):
            sentences[i].append(0) # pad zeros
            sentences_lengths[i].append(0) # pad zeros

#          print (sentences)
#          print (sentences_lengths)

        raw_sentence = np.asarray(sentences, dtype=np.int32)
#          print ("tensor is: ", raw_sentence)
        bos_sentence = np.concatenate((np.zeros((len(sentences), 1), dtype=np.int32), sentences), axis=1)
#          bos_sentence = np.delete(bos_sentence, max_length, 1)
#          print ("bos tensor is: ", bos_sentence)

        sentence_eos = np.concatenate((sentences, np.zeros((len(sentences), 1), dtype=np.int32)), axis=1)
#          sentence_eos = np.delete(sentence_eos, 0, 1)
#          print ("eos tensor is: ", sentence_eos)

        sentence_lengths = np.asarray(sentences_lengths, dtype=np.int32)
#          print ("length tensor is: ", sentences_lengths)

        result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/bos_sentences:0": bos_sentence, "Train/Model/sentences_eos:0": sentence_eos, "Train/Model/test_sentence_lengths:0": sentence_lengths})

        sentences = []
        sentences_lengths = []
        max_length = 0
        print (result)

#    x = np.array([[]], dtype=np.int32).astype(np.int32)
#    result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#    print (result)
#
#    x = np.array([[2]], dtype=np.int32).astype(np.int32)
#    result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#    print (result)
#
#    x = np.array([[2, 3]], dtype=np.int32).astype(np.int32)
#    result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#    print (result)
#
#    x = np.array([[2, 3, 1]], dtype=np.int32).astype(np.int32)
#    result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#    print (result)
#
#    x = np.array([[2, 3, 1, 0]], dtype=np.int32).astype(np.int32)
#    result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#    print (result)

if __name__ == "__main__":
  tf.app.run()
