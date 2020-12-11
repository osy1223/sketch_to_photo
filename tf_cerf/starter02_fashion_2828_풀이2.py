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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv1D, MaxPool1D, GlobalMaxPool1D
import numpy as np

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    #1. 데이터
    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)    # (60000, 28, 28)
    print(y_train.shape)    # (60000, )
    print(x_test.shape)     # (10000, 28, 28)
    print(y_test.shape)     # (10000, )

    # x_train = np.reshape(x_train, (60000, 28, 28, 1))
    # x_test = np.reshape(x_test, (10000, 28, 28, 1))

    # print(x_train[0])
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    # print(x_train[0])

    #2. 모델
    model = Sequential()
    model.add(Conv1D(256, input_shape=(28, 28), kernel_size=2,
                     padding="same", activation="relu"))
    # model.add(MaxPool1D())
    model.add(Conv1D(128, kernel_size=2, padding="valid", activation="relu"))

    model.add(Conv1D(128, kernel_size=2, activation="relu"))
    # model.add(MaxPool1D())
    model.add(GlobalMaxPool1D())

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()

    #3. 컴파일, 훈련
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=['acc'])
    model.fit(x_train, y_train, batch_size=256, epochs=100,
              validation_data=(x_test, y_test))

    #4. 평가, 예측

    # return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
