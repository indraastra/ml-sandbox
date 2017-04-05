import tensorflow as tf
import numpy as np
import os
import pickle
import time
import datetime
from tensorflow.contrib import learn
import csv

import data
from text_cnn import TextCNN

# Eval Parameters
tf.flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory from training run')

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')

passthrough_tokenizer = data.passthrough_tokenizer
vocab_path = os.path.join(FLAGS.checkpoint_dir, '..', 'vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
label_path = os.path.join(FLAGS.checkpoint_dir, '..', 'labels')
with open(label_path, 'rb') as label_file:
    labels = pickle.load(label_file)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()


def predict_cli(predict_fn):
    while True:
        user_input = input('> ')
        if not user_input:
            break
        user_tokens = data.tokenize_data(user_input)
        x = list(vocab_processor.transform([user_tokens]))
        prediction = predict_fn(x)
        print(labels[prediction])


with graph.as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        def predict_fn(x):
            return sess.run(predictions, {input_x: x,
                                          dropout_keep_prob: 1.0})[0]
        predict_cli(predict_fn)

