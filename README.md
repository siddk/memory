# Memory

Core repository for all research code centered around [Memory Networks](http://arxiv.org/pdf/1410.3916v11.pdf)
for question-answering. This directory is broken up in the following way:

1. BABI Tasks (/babi)
    - Replicate Question-Answering Code from [End-to-End Memory Networks](http://arxiv.org/pdf/1503.08895v5.pdf)
      on Facebook AI's corpus of [20 BABI Tasks](https://research.facebook.com/researchers/1543934539189348)

2. BABI LSTM Model (/rnn)
    - Experiment to see if a Recurrent Neural Network (LSTM) Model with question + story sentence fed
      together can match original BABI Memory Network Results.