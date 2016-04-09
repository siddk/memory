"""
util.py

Core util file, contains functions for loading and parsing tasks.
"""
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import re


def load_task(data_dir, task_id):
    """
    Load training and test data for the given task, return sets after parsing and vectorizing.

    :param data_dir: Path to task data (either 1K or 10K)
    :param task_id: Task to get data for.
    :return: Tuple of train data, test data
    """
    assert 0 < task_id < 21
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    train_file, test_file = sorted([f for f in files if 'qa%d_' % task_id in f])[::-1]
    train_data, test_data = get_stories(train_file), get_stories(test_file)
    return train_data, test_data


def get_stories(fp):
    """
    Given a path to a bAbI Task Dataset, open the file, and parse the stories, returning each
    example as a tuple of the following form: (context, question, answer)

    :param fp: Path to task dataset.
    :return: List of example tuples.
    """
    with open(fp, 'r') as f:
        lines = f.readlines()

    data, story = [], []
    for line in lines:
        line = line.lower()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []

        if '\t' in line:  # Question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            a = [a]

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            # Provide all the sub_stories
            sub_story = [x for x in story if x]

            data.append((sub_story, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)

    return data


def tokenize(sentence):
    """
    Return the tokens of the sentence as a list, including punctuation.

    :param sentence: Raw string containing complete sentence.
    :return: Tokenized sentence (as list).
    """
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]


def vectorize(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.

    :param data: Data of the form: (story, question, answer)
    :param word_idx: Dictionary mapping word to vocabulary index
    :param sentence_size: Maximum sentence length (to build vectors)
    :param memory_size: Maximum story length (to build vectors)
    :return: Vectorized data as tuple of story, question, answer
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)