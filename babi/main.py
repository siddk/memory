"""
main.py

Runner for bAbI Tasks MemN2N Memory Network. Specify which task to train model over, or whether
to run joint training over all 20 Tasks.

References:
    http://arxiv.org/pdf/1503.08895v5.pdf (End-to-End Memory Networks Sukhbaatar et. al)
    https://github.com/vinhkhuc/MemN2N-babi-python (Vinh Khuc's Python Implementation)
    https://github.com/facebook/MemNN/tree/master/MemN2N-babi-matlab (Original Matlab Code)
"""
from model.build_model import build_model
from model.test import test
from model.train import train, train_linear_start
from util.config import BabiConfig
from util.parser import parse_babi_task
import argparse
import glob
import numpy as np
import os
import random
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)  # for reproducing


def run_task(data_directory, task_id):
    """
    Parse data, build model, and run training and testing for a single task.

    :param data_directory: Path to train and test data.
    :param task_id: Task to evaluate
    """
    print("Train and test for task %d ..." % task_id)

    # Parse data
    train_files = glob.glob('%s/qa%d_*_train.txt' % (data_directory, task_id))
    test_files = glob.glob('%s/qa%d_*_test.txt' % (data_directory, task_id))

    dictionary = {"nil": 0}

    # Story shape: (SENTENCE_SIZE, STORY_SIZE, NUM_STORIES)
    # Questions shape: (14 (see parser.py), NUM_SAMPLES)
    # QStory shape: (SENTENCE_SIZE, NUM_SAMPLES)
    train_story, train_questions, train_qstory = parse_babi_task(train_files, dictionary, False)
    test_story, test_questions, test_qstory = parse_babi_task(test_files, dictionary, False)

    general_config = BabiConfig(train_story, train_questions, dictionary)

    memory, model, loss = build_model(general_config)

    if general_config.linear_start:
        train_linear_start(train_story, train_questions, train_qstory, memory, model, loss,
                           general_config)
    else:
        train(train_story, train_questions, train_qstory, memory, model, loss, general_config)

    test(test_story, test_questions, test_qstory, memory, model, loss, general_config)


def run_all_tasks(data_directory):
    """
    Train and test for all tasks.

    :param data_directory Path to data directory
    """
    print("Training and testing for all tasks ...")
    for t in range(20):
        run_task(data_directory, task_id=t + 1)

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
    if args.all_tasks:
        run_all_tasks(data_dir)
    elif args.joint_tasks:
        pass
        # run_joint_tasks(data_dir)
    else:
        run_task(data_dir, task_id=args.task)