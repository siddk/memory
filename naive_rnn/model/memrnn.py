"""
memrnn.py

Core model file, defines keras LSTM Model for bAbI RNN.
"""
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Embedding, Flatten, LSTM, Reshape
from keras.models import Sequential
import numpy as np


class MemLSTM:
    def __init__(self, train_x, train_y, embedding_size=100, memory_size=100):
        """
        Instantiate a MemLSTM Model with the training data for stories, questions, and answers.

        :param train_x: Story + Query tensor with shape (N, STORY_SIZE, SENTENCE_SIZE)
        :param train_y: Answer tensor with one-hot shape (N, STORY_SIZE, VOCAB_SIZE)
        """
        self.train_x, self.train_y = train_x, train_y
        self.N, self.STORY_SIZE, self.SENTENCE_SIZE = train_x.shape
        _, _, self.V = train_y.shape
        self.EMBEDDING_SIZE, self.MEMORY_SIZE = embedding_size, memory_size

        self.model = self.build_model2()

    def build_model(self):
        """
        Build Keras Model
        """
        model = Sequential()
        # Input Shape: (_, STORY_SIZE, SENTENCE_SIZE)
        model.add(TimeDistributed(Embedding(input_dim=self.V, output_dim=self.EMBEDDING_SIZE,
                                            input_length=self.SENTENCE_SIZE),
                                  input_shape=(self.STORY_SIZE, self.SENTENCE_SIZE)))
        # Shape: (_, STORY_SIZE, SENTENCE_SIZE, EMBEDDING_SIZE)
        model.add(TimeDistributed(Flatten()))
        # Shape: (_, STORY_SIZE, SENTENCE_SIZE * EMBEDDING_SIZE)
        model.add(LSTM(self.MEMORY_SIZE, return_sequences=True))
        # Shape: (_, STORY_SIZE, MEMORY_SIZE)
        model.add(TimeDistributed(Dense(self.V, activation='softmax')))
        # Shape: (_, STORY_SIZE, VOCAB_SIZE) --> Softmax

        return model

    def build_model2(self):
        """
        Build Keras model of MemLSTM
        """
        self.train_x = np.reshape(self.train_x, (self.N, self.STORY_SIZE * self.SENTENCE_SIZE))
        model = Sequential()
        # Shape: (N, STORY_SIZE * SENTENCE_SIZE)
        model.add(Embedding(input_dim=self.V, output_dim=self.EMBEDDING_SIZE,
                            input_length=self.STORY_SIZE * self.SENTENCE_SIZE, dropout=0.3))
        # Shape: (N, STORY_SIZE * SENTENCE_SIZE, EMBEDDING_SIZE)
        model.add(Reshape((self.STORY_SIZE, self.SENTENCE_SIZE, self.EMBEDDING_SIZE)))
        # Shape: (N, STORY_SIZE, SENTENCE_SIZE, EMBEDDING_SIZE)
        model.add(TimeDistributed(Flatten()))
        # Shape: (N, STORY_SIZE, SENTENCE_SIZE * EMBEDDING_SIZE)
        model.add(LSTM(self.MEMORY_SIZE, return_sequences=True))
        # Shape: (N, STORY_SIZE, MEMORY_SIZE)
        model.add(TimeDistributed(Dense(self.V, activation='softmax')))
        # Shape: (N, STORY_SIZE, VOCAB_SIZE) --> Softmax

        return model

    def train(self):
        """
        Compile and fit the model.
        """
        self.model.compile(optimizer='Adam', loss='mse',
                           metrics=['accuracy'])
        self.model.fit(self.train_x, self.train_y, batch_size=32, nb_epoch=100,
                       validation_split=0.05)

