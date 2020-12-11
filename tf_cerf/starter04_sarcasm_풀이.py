# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    with open('sarcasm.json', 'r') as f:
        datastore = json.load(f)

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE    #1. 데이터

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentenses = sentences[0:training_size]
    test_sentenses = sentences[training_size:]
    training_labels = labels[0:training_size]
    test_labels = labels[training_size:]

    # print(training_sentenses)
    print(training_sentenses[0])
    print(training_labels[0])
    print(training_sentenses[1111])
    print(training_labels[1111])

    # 넘파이로 변경
    training_sentenses = np.array(training_sentenses)
    test_sentenses = np.array(test_sentenses)
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    # 토크나이저
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentenses)

    word_index = tokenizer.word_index
    print(word_index)

    training_sequences = tokenizer.texts_to_sequences(training_sentenses)
    print(training_sequences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                    padding=padding_type, truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_sentenses)
    test_padded = pad_sequences(test_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

    #2. 모델구성
    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
        tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
        tf.keras.layers.Conv1D(64, 2, activation='relu'),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])



    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(training_padded, training_labels, epochs=20,
              validation_data=(test_padded, test_labels), verbose=1)

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
