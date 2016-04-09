"""
babi.py

Main file for training and evaluating Memory Network on set of bAbI tasks. Specify in the command
line whether to run the independent or joint training, and if independent, which task to train
on.

References:
    http://arxiv.org/abs/1503.08895 (End-to-End Memory Networks Paper)
    https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py (Keras Implementation)
    https://github.com/domluna/memn2n (TensorFlow Implementation)
"""
from itertools import chain
from util.util import *
import argparse
import numpy as np

MEMORY_SIZE = 50  # Maximum Number of sentences to store in Memory
DATA_DIR = "data/en"
DATA_DIR_10K = "data/en-10k"

if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Set Parameters for bAbI Memory Network Training')
    parser.add_argument('task_id', metavar='TASK_ID', type=int, help="Task to train and evaluate")
    parser.add_argument('--joint', dest='joint', action='store_const', const=True, default=False,
                        help='Boolean Flag to specify whether training over independent task (' +
                             'default), or all tasks together')
    parser.add_argument('--10k', dest='large', action='store_const', const=True, default=False,
                        help='Boolean Flag to specify whether to use the 1000 example training' +
                             'set (default), or the 10000 example training set')
    args = parser.parse_args()
    task, use_joint, use_large = args.task_id, args.joint, args.large
    train, test = None, None

    # Independent Task Training
    if not use_joint:
        print "Started Task:", task
        if use_large:
            print "Using dataset of 10K Examples"
        else:
            print "Using dataset of 1K Examples"

        data_dir = DATA_DIR_10K if use_large else DATA_DIR
        train, test = load_task(data_dir, task)

    # Joint Task Training
    else:
        pass

    # Build up vectorized versions of data
    vocab = sorted(reduce(lambda x, y: x | y,
                         (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    story_max = max(map(len, (s for s, _, _ in train + test)))
    mean_story_size = int(np.mean(map(len, (s for s, _, _ in train + test))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train + test)))
    query_max = max(map(len, (q for _, q, _ in train + test)))
    memory_size = min(MEMORY_SIZE, story_max)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_max, sentence_size)  # for the position

    # Output current model parameters
    print '-'
    print 'Vocab size:', vocab_size, 'unique words'
    print 'Story max length:', story_max, 'sentences'
    print 'Average story length:', mean_story_size, 'sentences'
    print 'Query max length:', query_max, 'words'
    print "Longest sentence length", sentence_size, 'words'
    print 'Number of training stories:', len(train)
    print 'Number of test stories:', len(test)
    print '-'
    print 'Here\'s what a "story" tuple looks like (input, query, answer):'
    print train[0]
    print '-'
    print 'Vectorizing the word sequences...'

    sent_s, mem_s = sentence_size, memory_size
    stories_train, questions_train, answers_train = vectorize(train, word_idx, sent_s, mem_s)
    stories_test, questions_test, answers_test = vectorize(test, word_idx, sent_s, mem_s)

    print '-'
    print 'Stories: integer tensor of shape (samples, memory_size, max_length)'
    print 'stories_train shape:', stories_train.shape
    print 'stories_test shape:', stories_test.shape
    print '-'
    print 'Questions: integer tensor of shape (samples, max_length)'
    print 'questions_train shape:', questions_train.shape
    print 'questions_test shape:', questions_test.shape
    print '-'
    print 'Answers: binary (1 or 0) tensor of shape (samples, vocab_size)'
    print 'answers_train shape:', answers_train.shape
    print 'answers_test shape:', answers_test.shape
    print '-'
    print 'Compiling...'

    # Build Memory Network


