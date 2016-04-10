"""
test.py

Function definitions for all test code, performs forward propagation on the model, evaluates
the results on the test data.
"""
import math
import numpy as np


def test(test_story, test_questions, test_qstory, memory, model, loss, general_config):
    """
    Run Memory Network Testing

    :param test_story: Tensor of Stories (Shape: (SENTENCE_SIZE, STORY_SIZE, NUM_STORIES))
    :param test_questions: Tensor of Questions (Shape: (14 (see parser.py), NUM_SAMPLES))
    :param test_qstory: Tensor of Q Indices within story (Shape: (SENTENCE_SIZE, NUM_SAMPLES))
    :param memory: Memory (Story) Network
    :param model: Sequential Query Network
    :param loss: Loss Network
    :param general_config: bAbI Configuration
    """
    total_test_err = 0.
    total_test_num = 0

    nhops = general_config.nhops
    train_config = general_config.train_config
    batch_size = general_config.batch_size
    dictionary = general_config.dictionary
    enable_time = general_config.enable_time

    max_words = train_config["max_words"] \
        if not enable_time else train_config["max_words"] - 1

    for k in range(int(math.floor(test_questions.shape[1] / batch_size))):
        batch = np.arange(k * batch_size, (k + 1) * batch_size)

        input_data = np.zeros((max_words, batch_size), np.float32)
        target_data = test_questions[2, batch]

        input_data[:] = dictionary["nil"]
        memory[0].data[:] = dictionary["nil"]

        for b in range(batch_size):
            d = test_story[:, :(1 + test_questions[1, batch[b]]), test_questions[0, batch[b]]]

            offset = max(0, d.shape[1] - train_config["sz"])
            d = d[:, offset:]

            memory[0].data[:d.shape[0], :d.shape[1], b] = d

            if enable_time:
                memory[0].data[-1, :d.shape[1], b] = np.arange(d.shape[1])[::-1] + len(
                    dictionary)  # time words

            input_data[:test_qstory.shape[0], b] = test_qstory[:, batch[b]]

        for i in range(1, nhops):
            memory[i].data = memory[0].data

        out = model.fprop(input_data)
        # cost = loss.fprop(out, target_data)
        total_test_err += loss.get_error(out, target_data)
        total_test_num += batch_size

    test_error = total_test_err / total_test_num
    print("Test error: %f" % test_error)