from collections import defaultdict
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            if clean_line: yield clean_line

def load_data(data_dir, per_lang_file_limit=10, train_test_split=0.8):
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    for lang in os.listdir(data_dir):
        lang_dir = os.path.join(data_dir, lang)
        # Read all lines for `per_lang_file_limit` files in that language.
        for file_name in os.listdir(lang_dir)[:per_lang_file_limit]:
            file_path = os.path.join(lang_dir, file_name)
            lines = list(load_lines(file_path))
            # Send split% to training, (100-split)% to test.
            split = int(train_test_split * len(lines))
            train_data[lang].extend(lines[:split])
            test_data[lang].extend(lines[split:])
    return train_data, test_data
