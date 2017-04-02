###
# Based on https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
# Modified to use TF batching.
###

import tensorflow as tf
import numpy as np
import os
import time
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn

import data
from text_cnn import TextCNN

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float('train_val_split', .1,
              'Percentage of the training data to use for validation')
tf.flags.DEFINE_string('data_directory', './data',
               'Data source organized as one directory of files per label.')
tf.flags.DEFINE_integer('min_word_frequency', 25, 'Minimum number of occurrences needed to include a word from the data (default: 25)')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 128, 'Dimensionality of character embedding (default: 128)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda (default: 0.0)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
tf.flags.DEFINE_integer('num_epochs', 3, 'Number of training epochs (default: 3)')
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on val set after this many steps (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps (default: 100)')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5)')
# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('Parameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print()


# Data Preparation
# ==================================================

# Load data.
print('Loading data:')
text_data = data.load_data(FLAGS.data_directory, per_label_file_limit=100)
x_text, y, labels = data.label_data(text_data)
print('Num labels: {}'.format(len(labels)))
print()

# Build vocabulary.
def passthrough_tokenizer(iter):
    for tokens in iter:
        yield tokens

max_document_length = max(len(l) for l in x_text)
print('Max document length: {:d}'.format(max_document_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length,
        tokenizer_fn=passthrough_tokenizer,
        min_frequency=FLAGS.min_word_frequency)
x = np.array(list(vocab_processor.fit_transform(x_text)))
y = np.array(y).reshape((len(y), 1))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

# Shuffle data.
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set.
val_sample_index = -int(FLAGS.train_val_split * len(y))
x_train, x_val = x_shuffled[:val_sample_index], x_shuffled[val_sample_index:]
y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=len(labels),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Val summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def val_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a val set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, val_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        x_train_single, y_train_single = tf.train.slice_input_producer(
                [x_train, y_train],
                num_epochs=FLAGS.num_epochs,
                shuffle=False)
        x_train_batches, y_train_batches = tf.train.shuffle_batch(
                [x_train_single, y_train_single],
                batch_size=FLAGS.batch_size,
                capacity=2000,
                min_after_dequeue=1000)

        # Initialize all variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Training loop.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                x_train_batch, y_train_batch = sess.run([x_train_batches,
                                                         y_train_batches])
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('\nEvaluation:')
                    val_step(x_val, y_val, writer=val_summary_writer)
                    print()
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}\n'.format(path))
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs.' % (FLAGS.num_epochs,))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

