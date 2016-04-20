"""
memrnn.py

Core model file, defines keras LSTM Model for bAbI RNN.
"""
from keras.layers import Dense, Dropout, Embedding, GRU, LSTM, Merge, RepeatVector
from keras.models import Sequential
import numpy as np


class MemLSTM:
    def __init__(self, train_s, train_q, train_a, embedding_size=50):
        """
        Instantiate a MemLSTM Model with the training data for stories, questions, and answers.

        :param train_s: Story tensor with shape (N, STORY_LEN, SENTENCE_LEN)
        :param train_q: Query tensor with shape (N, SENTENCE_LEN)
        :param train_a: Answer tensor with one-hot shape (N, VOCAB_SIZE)
        """
        self.train_s, self.train_q, self.train_a = train_s, train_q, train_a
        self.N, self.STORY_SIZE = train_s.shape
        _, self.QUERY_SIZE = train_q.shape
        _, self.V = train_a.shape
        self.EMBEDDING_SIZE = embedding_size

        self.model = self.build_model()

    def build_model(self):
        """
        Build Keras model of MemLSTM
        """
        story_rnn = Sequential()
        # Shape: (N, STORY_SIZE)
        story_rnn.add(Embedding(input_dim=self.V, output_dim=self.EMBEDDING_SIZE,
                                input_length=self.STORY_SIZE))
        story_rnn.add(Dropout(0.3))
        # Shape: (N, STORY_SIZE, EMBEDDING_SIZE)

        query_rnn = Sequential()
        # Shape: (N, QUERY_SIZE)
        query_rnn.add(Embedding(input_dim=self.V, output_dim=self.EMBEDDING_SIZE,
                                input_length=self.QUERY_SIZE))
        query_rnn.add(Dropout(0.3))
        # Shape: (N, QUERY_SIZE, EMBEDDING_SIZE)
        query_rnn.add(GRU(self.EMBEDDING_SIZE, return_sequences=False))
        # Shape: (N, EMBEDDING_SIZE)
        query_rnn.add(RepeatVector(self.STORY_SIZE))
        # Shape: (N, STORY_SIZE, EMBEDDING_SIZE)

        model = Sequential()
        # Shape 1: (N, STORY_SIZE, EMBEDDING_SIZE) Shape 2: (N, STORY_SIZE, EMBEDDING_SIZE)
        model.add(Merge([story_rnn, query_rnn], mode='sum'))
        # Shape: (N, STORY_SIZE, EMBEDDING_SIZE)
        model.add(GRU(self.EMBEDDING_SIZE, return_sequences=False))
        model.add(Dropout(0.3))
        # Shape: (N, EMBEDDING_SIZE)
        model.add(Dense(self.V, activation='softmax'))
        # Shape: (N, VOCAB_SIZE)
        return model

    def train(self):
        """
        Compile and fit the model.
        """
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit([self.train_s, self.train_q], self.train_a, batch_size=32, nb_epoch=100,
                       validation_split=0.05)
