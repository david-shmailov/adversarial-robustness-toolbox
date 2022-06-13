#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import pickle
from glob import glob
# from art.attacks.evasion import HopSkipJump
from hop_skip_jump import HopSkipJump
from art.estimators.classification import KerasClassifier
from art.defences.preprocessor import *
from art.utils import load_dataset
import tensorflow as tf
import argparse
from os.path import exists
import multiprocessing


def main(func1, test, args, preprocessor=None):
    epochs = int(args.epoch) if args.epoch else 5
    accuracy_before_attack = 0
    path_for_results = './results/'
    log_name = f"{path_for_results}{func1}_{test}_results_log.txt"

    tf.compat.v1.disable_eager_execution()
    # Read MNIST dataset
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))

    # Create Keras convolutional neural network - basic architecture from Keras examples
    # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=func1, input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation=func1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=func1))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    classifier = KerasClassifier(model=model, clip_values=(min_, max_), use_logits=False, preprocessing_defences=preprocessor)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))
    accuracy_before_attack = acc * 100

    # Craft adversarial samples with FGSM
    if args.d:
        adv_crafter = HopSkipJump(classifier, log_file=log_name, max_eval=1, init_eval=1, max_iter=1)
        # single_image = x_test
        x_test_adv = adv_crafter.generate(x=x_test)
    else:
        adv_crafter = HopSkipJump(classifier, log_file=log_name)
        x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("function: {}, preprocessor: {}".format(func1, test))
    print("\nTest accuracy on adversarial sample: %.2f%% " % (acc * 100))
    accuracy_after_attack = acc * 100
    with open(log_name, 'a') as log_file:
        result_before = "Test accuracy: %.2f%%\n" % accuracy_before_attack
        result_after = "Test accuracy on adversarial sample: %.2f%%\n" % accuracy_after_attack
        log_file.write(result_before)
        log_file.write(result_after)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help="debug, very short hop_skip run")
    parser.add_argument('-all', action='store_true', help="Run all activation function tests in parallel")
    parser.add_argument('-epoch', action="store", help = "number of epochs for model. default 5")
    args = parser.parse_args()

    activation_functions = [
        'relu',
        # 'gelu',
        'elu',
        # 'selu',
        # 'tanh',
        # 'sigmoid',
    ]
    preprocessors = {
        'gaussian_train':GaussianAugmentation(augmentation=False, apply_fit=True, apply_predict=False),
        'gaussian_predict':GaussianAugmentation(augmentation=False, apply_fit=False, apply_predict=True),
        'gaussian_both':GaussianAugmentation(augmentation=False, apply_fit=True, apply_predict=True),
        'spacial_smooth_train':SpatialSmoothing( apply_fit=True, apply_predict=False),
        'spacial_smooth_predict': SpatialSmoothing( apply_fit=False, apply_predict=True),
        'spacial_smooth_both': SpatialSmoothing( apply_fit=True, apply_predict=True),
        'variance_min_train': TotalVarMin( apply_fit=True, apply_predict=False),
        'variance_min_predict': TotalVarMin( apply_fit=False, apply_predict=True),
        'variance_min_both': TotalVarMin( apply_fit=True, apply_predict=True),
        'label_smooth_train':LabelSmoothing( apply_fit=True, apply_predict=False),
        'label_smooth_predict': LabelSmoothing(apply_fit=False, apply_predict=True),
        'label_smooth_both': LabelSmoothing(apply_fit=True, apply_predict=True),
    }

    if not args.all:
        main('elu', 'elu', args)
    else:
        with open('simulation_log.txt','w') as log:
            for func in activation_functions:
                for test, preprocessor in preprocessors.items():
                    try:
                        print(f"________________function: {func}, preprocessor: {test}________________")
                        main(func,test,args,preprocessor)
                    except Exception as e:
                        log.write(f'test running {func} with {test} failed due to:\n{str(e)}\n')
