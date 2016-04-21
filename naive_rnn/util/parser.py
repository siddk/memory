"""
parser.py

Read data, tokenize and vectorize, storing stories, questions, and answers into numpy arrays.
"""
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


def get_stories(filename, max_length=None):
    """
    Given a file name, read the file, retrieve the stories, and then convert the sentences into a
    single story. If max_length is supplied, any stories longer than max_length tokens will be
    discarded.

    :param f: Python file object.
    :return Tokenized and parsed data.
    """
    with open(filename, 'r') as f:
        data = parse_stories(f.readlines())
        data = [(story, q, answer) for story, q, answer in data]
        data = [(story, q, answer) for story, q, answer in data if not max_length or
                len(story) < max_length]
        return data


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.

    :param sent: Sentence to tokenize (str)
    :return: List of tokens.
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines):
    """
    Parse stories provided in the bAbi tasks format.

    :param lines: Lines in the task file.
    :return: Parsed data
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            # Provide all the sub-stories
            substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    """
    Turn story into matrix/tensor form.

    :param data: Tokenized and parsed data.
    :param word_idx: Vocabulary Dictionary.
    :param story_maxlen: Maximum story length.
    :param query_maxlen: Maximum question length.
    :return: Vectorized data.
    """
    stories, queries, answers = [], [], []
    for story, query, answer in data:
        s = []
        for line in story:
            s.append([word_idx[w] for w in line])
        q = [word_idx[w] for w in query]
        a = np.zeros(len(word_idx) + 1)   # Let's not forget that index 0 is reserved
        a[word_idx[answer]] = 1

        assert len(q) <= query_maxlen
        for i in range(len(s)):
            assert len(s[i]) <= query_maxlen
            while len(s[i]) < query_maxlen:
                s[i].append(0)

        if len(s) > story_maxlen:
            s = s[::-1][:50][::-1]

        while len(s) < story_maxlen:
            s.append([0 for _ in range(query_maxlen)])

        while len(q) < query_maxlen:
            q.append(0)

        stories.append(s)
        queries.append(q)
        answers.append(a)

    return np.array(stories, dtype='int'), np.array(queries, dtype='int'), np.array(answers, dtype='int')