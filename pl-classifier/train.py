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
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev set after this many steps (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps (default: 100)')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5)')
# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')


# Data Preparation
# ==================================================

# Load data.
print('Loading data:')
text_data = data.load_data(FLAGS.data_directory, per_label_file_limit=10)
x_text, y, labels = data.label_data(text_data)
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
y = np.array(y)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

#x_train, y_train, x_test, y_test = data.split_data(FLAGS.train_val_split, shuffle=True)
# Shuffle data.
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set.
dev_sample_index = -int(FLAGS.train_val_split * len(y))
x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))


# Send split% to training, (100-split)% to test.
#split = int(train_test_split * len(lines))
#train_data[lang].extend(lines[:split])
#test_data[lang].extend(lines[split:])

