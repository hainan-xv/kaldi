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

FLAGS = flags.FLAGS

def main(_):
  if not FLAGS.model_path:
    raise ValueError("Must set --model-path to the trained RNNLM model")

  wordmap = _build_vocab(FLAGS.wordmap_path)

  oos_id = wordmap["<oos>"]



  with tf.Session() as session:
    # restore the model
    saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
    saver.restore(session, FLAGS.model_path)

#    for i in range(0, 10000):
##      for j in range(0, 10000):
#      x = np.array([[0, i]], dtype=np.int32).astype(np.int32)
#      result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#      print (result)
#    return

    with open(FLAGS.text_path, "r") as f:
      for line in f:
        word_ids = []
        for word in line.split():
          if word in wordmap:
            word_ids.append(wordmap[word])
          else:
            word_ids.append(oos_id)

#        print (word_ids)
        x = np.array(np.asarray([word_ids]), dtype=np.int32).astype(np.int32)
        result = session.run("Train/Model/nbest_out:0", feed_dict={"Train/Model/test_sentences:0": x})
#        print (line, result)
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
