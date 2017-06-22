#! /usr/bin/env python

# original script from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/eval.py

import tensorflow as tf
import numpy as np
import os, pickle
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

from prepare_datafile import load_data_and_labels, prepare_test_data,  load_bin_vec, get_W, batch_iter

# Parameters
# ==================================================

# Data loading params

#tf.flags.DEFINE_string("data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "../data/GoogleNews-vectors-negative300.bin", "w2v file.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1497871399/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
project_folder = os.path.abspath(os.path.join(os.path.abspath(os.curdir), os.pardir))
test_datafile = os.path.join(project_folder,"data/datafile_test.txt")
vocab = prepare_test_data(test_datafile)
vocab = set([word.lower() for word in vocab if not word.isalnum()]) # todo fix a better way later
x_text, y_one_hot = load_data_and_labels(vocab,datafile=test_datafile)

print("Loading w2v...")
dim, word_vecs = load_bin_vec(vocab) # fname=FLAGS.w2v_file
print("Loading idx map...")
W, word_idx_map = get_W(word_vecs)
embeddings = pickle.load("embeddings.pck")
#embeddings = W[1:]
print("Starting ...")

#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))

x_test = np.array(list([W[word_idx_map[word]] for word in x_text ]))
y_test = np.array(y_one_hot)


print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        embedding_placeholder = graph.get_operation_by_name("embedding/Placeholder").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0,
                                                       embedding_placeholder: embeddings })
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
