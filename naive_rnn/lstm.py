"""
lstm.py

Core implementation for bAbI RNN -> A Model that feeds each query with each sentence, to build
a better attention mechanism over the story sentences, to better contribute to the answer.
"""
from model.memrnn import MemLSTM
from util.parser import *
import argparse
import glob
import os
import sys

EMBEDDING_SIZE = 50


def run_task(data_directory, task):
    """
    Given the data directory, and the task in question, build, train, and evaluate LSTM RNN
    Model.

    :param data_directory: Path to training/testing data.
    :param task: Task to train on.
    """
    assert 0 < task < 21
    train_data = glob.glob('%s/qa%d_*_train.txt' % (data_directory, task))[0]
    test_data = glob.glob('%s/qa%d_*_test.txt' % (data_directory, task))[0]

    train, test = get_stories(train_data), get_stories(test_data)
    vocab = sorted(reduce(lambda x, y: x | y,
                          (set(story + q + [answer]) for story, q, answer in train + test)))

    vocab_size = len(vocab) + 1
    vocab = dict((c, i + 1) for i, c in enumerate(vocab))
    s_max = max(map(len, (x for x, _, _ in train + test)))
    q_max = max(map(len, (x for _, x, _ in train + test)))

    s_train, q_train, a_train = vectorize_stories(train, vocab, s_max, q_max)
    s_test, q_test, a_test = vectorize_stories(test, vocab, s_max, q_max)

    model = MemLSTM(s_train, q_train, a_train)
    model.train()
    loss, acc = model.model.evaluate([s_test, q_test], a_test, batch_size=32)
    print "Testing Accuracy", acc
    return model, loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", default="data/en",
                        help="path to dataset directory (default: %(default)s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--task", default="1", type=int,
                       help="train and test for a single task (default: %(default)s)")
    group.add_argument("-a", "--all-tasks", action="store_true",
                       help="train and test for all tasks (one by one) (default: %(default)s)")
    group.add_argument("-j", "--joint-tasks", action="store_true",
                       help="train and test for all tasks (all together) (default: %(default)s)")
    args = parser.parse_args()

    # Check if data is available
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print("The data directory '%s' does not exist. Please download it first." % data_dir)
        sys.exit(1)

    print("Using data from %s" % args.data_dir)

    # Run model on given task
    mem_lstm, l, a = run_task(data_dir, task=args.task)