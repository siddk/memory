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

EMBEDDING_SIZE = 20


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
                          (set(reduce(lambda x, y: x + y, story) + q + [answer]) for
                           story, q, answer in train + test)))

    vocab = dict((c, i + 1) for i, c in enumerate(vocab))
    story_max = max(map(len, (x for x, _, _ in train + test)))
    sentence_max = 0
    for a, b, _ in train + test:
        if len(b) > sentence_max:
            sentence_max = len(b)
        for s in a:
            if len(s) > sentence_max:
                sentence_max = len(s)

    s_train, q_train, a_train = vectorize_stories(train, vocab, story_max, sentence_max)
    s_test, q_test, a_test = vectorize_stories(test, vocab, story_max, sentence_max)

    # Build new concatenated training sets
    qtrain_repeat = np.tile(q_train, story_max).reshape(s_train.shape)
    qtest_repeat = np.tile(q_test, story_max).reshape(s_test.shape)

    atrain_repeat = np.tile(a_train, story_max).reshape(s_train.shape[:-1] + a_train.shape[-1:])
    atest_repeat = np.tile(a_test, story_max).reshape(s_test.shape[:-1] + a_test.shape[-1:])

    train_x = np.concatenate((s_train, qtrain_repeat), axis=-1)
    train_y = atrain_repeat

    test_x = np.concatenate((s_test, qtest_repeat), axis=-1)
    n, st, ss = test_x.shape
    test_x = np.reshape(test_x, (n, st * ss))
    test_y = atest_repeat

    # Build, train, and evaluate model
    model = MemLSTM(train_x, train_y)
    model.train()
    out = model.model.predict(test_x, batch_size=32)
    return model, out, vocab, test_y


def evaluate_predictions(p, vocab, test_data):
    """
    Look at predictions, and evaluate last element to evaluate test accuracy.

    :param p:
    :param vocab:
    :param test_data:
    """
    voc = {i: c for c, i in vocab.items()}
    import pickle
    with open('task_1_predictions.pik', 'w') as f:
        pickle.dump(p, f)

    correct, total = 0, 0
    for index in range(len(p)):
        preds = get_prediction(p, index, voc)
        answer = voc[test_data[index].argmax()]
        print "ANSWER:", index, preds[-1], answer
        if preds[-1] == answer:
            correct += 1
        total += 1

    return float(correct) / float(total)


def get_prediction(p, index, vocab):
    curr = p[index]
    preds = []
    for i in curr:
        pred = vocab[i.argmax()]
        preds.append(pred)
        print pred
    return preds


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
    mem_lstm, predictions, v, y = run_task(data_dir, task=args.task)
    acc = evaluate_predictions(predictions, v, y)
    print "Accuracy", acc
