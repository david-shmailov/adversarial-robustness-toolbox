# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import pickle
from glob import glob
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
import tensorflow as tf
from os.path import exists
from multiprocessing import Pool


def main(func):
    # drive_path = "/content/drive/My Drive/Colab Notebooks/"
    force_train = True
    log_name = "{}_results_log.txt".format(func)
    classifier_file = "{}_trained_classifier".format(func)
    if not glob(classifier_file) or force_train:
        tf.compat.v1.disable_eager_execution()
        # Read MNIST dataset
        (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))

        # Create Keras convolutional neural network - basic architecture from Keras examples
        # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation=func, input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation=func))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=func))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        classifier = KerasClassifier(model=model, clip_values=(min_, max_), use_logits=False)
        classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("\nTest accuracy: %.2f%%" % (acc * 100))
        pickle.dump(classifier, open(classifier_file, "wb"))
    else:
        classifier = pickle.load(open(classifier_file, "rb"))
    # Craft adversarial samples with FGSM

    adv_crafter = HopSkipJump(classifier, log_file=log_name)
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("function: " + func)
    print("\nTest accuracy on adversarial sample: %.2f%% " % (acc * 100))
    return acc * 100


retrain = True
activation_functions = ['gelu',
                        'elu',
                        'selu',
                        'tanh',
                        ]

if __name__ == '__main__':
    main('elu')
    #with Pool(2) as p:
    #    print(p.map(main, activation_functions))