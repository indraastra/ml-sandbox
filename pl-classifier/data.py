from collections import defaultdict
import os
import re

###
# Based on https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
###

def tokenize_data(string):
    """
    Tokenization/string cleaning for code.
    """
    string = re.sub(r"([^\w\d\s])", r" \1 ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(' ')


def load_lines(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        for line in file:
            clean_line = tokenize_data(line)
            if clean_line: yield clean_line


def load_data(data_dir, per_label_file_limit=1000):
    """
    Loads file lines into a dict keyed by language.
    """
    data = defaultdict(list)
    for lang in os.listdir(data_dir):
        print('Loading language: {}'.format(lang))
        lang_dir = os.path.join(data_dir, lang)
        # Read all lines for `per_label_file_limit` files of that label.
        for file_name in os.listdir(lang_dir)[:per_label_file_limit]:
            file_path = os.path.join(lang_dir, file_name)
            data[lang].extend(load_lines(file_path))
    return data


def label_data(data_dict):
    all_x = []
    all_y = []
    labels = {label: idx for (idx, label) in enumerate(data_dict.keys())}
    for label, data in data_dict.items():
        for datum in data:
            all_x.append(datum)
            all_y.append(labels[label])
    return all_x, all_y, sorted(labels.keys(), key=lambda l: labels[l])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
