"""
config.py

Configuration file containing config classes for both versions of the bAbI Task Model:
    1) Single Task (train model per task)
    2) Joint Tasks (train model on all tasks simultaneously)
"""
import numpy as np


class BabiConfig(object):
    def __init__(self, train_story, train_questions, dictionary):
        """
        Initialize Babi Configuration, with the following parameters:

        :param train_story: Tensor of Stories (Shape: (SENTENCE_SIZE, STORY_SIZE, NUM_STORIES))
        :param train_questions: Tensor of Questions (Shape: (14 (see parser.py), NUM_SAMPLES))
        :param dictionary: Dictionary Mapping words to integers.
        """
        self.dictionary = dictionary
        self.batch_size = 32
        self.nhops = 3
        self.nepochs = 100
        self.lrate_decay_step = 25  # reduce learning rate by half every 25 epochs

        # Use 10% of training data for validation
        nb_questions = train_questions.shape[1]
        nb_train_questions = int(nb_questions * 0.9)

        self.train_range = np.array(range(nb_train_questions))
        self.val_range = np.array(range(nb_train_questions, nb_questions))
        self.enable_time = True    # Add time embeddings
        self.use_bow = False       # Use Bag-of-Words instead of Position-Encoding
        self.linear_start = True
        self.share_type = 1        # 1: adjacent, 2: layer-wise weight tying
        self.randomize_time = 0.1  # Amount of noise injected into time index
        self.add_proj = False      # Add linear layer between internal states
        self.add_nonlin = False    # Add non-linearity to internal states

        if self.linear_start:
            self.ls_nepochs = 20
            self.ls_lrate_decay_step = 21
            self.ls_init_lrate = 0.01 / 2

        # Training configuration
        self.train_config = {
            "init_lrate": 0.01,
            "max_grad_norm": 40,
            "in_dim": 20,
            "out_dim": 20,
            "sz": min(50, train_story.shape[1]),  # Number of sentences
            "voc_sz": len(self.dictionary),
            "bsz": self.batch_size,
            "max_words": len(train_story),
            "weight": None
        }

        if self.linear_start:
            self.train_config["init_lrate"] = 0.01 / 2

        if self.enable_time:
            self.train_config.update({
                "voc_sz": self.train_config["voc_sz"] + self.train_config["sz"],
                "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
            })
