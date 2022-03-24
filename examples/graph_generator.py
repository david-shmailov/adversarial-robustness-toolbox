import argparse
from glob import glob
import sys
import re
from os.path import isfile, isdir
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline

size_of_data_set = 10000
accuracies_after = {
    'relu': 0.8,
    'gelu': 0,
    'elu' : 10.36,
    'selu': 0,
    'tanh': 10.43,
    'sigmoid': 0,
}
accuracies_before = {
    'relu': 98.97,
    'gelu': 98.92,
    'elu' : 98.74,
    'selu': 98.43,
    'tanh': 98.62,
    'sigmoid': 96.94,
}

class GraphGenerator:
    def __init__(self, args):
        self.files_data = {}
        self.files_data_including_failed = {}
        self.percentage_of_failures = {}
        self.args = args

    def run(self):
        if isfile(self.args.i):
            path = glob(self.args.i)[0]
            self.read_results(path)
        elif isdir(self.args.i):
            files = glob(self.args.i + "*_results_log.txt")
            for file in files:
                self.read_results(file)
        self.create_graphs()
        #self.create_box_plot()
        #self.create_percentage_bar()
        #self.network_accuracy_bar()

    def create_graphs(self):
        plt.figure(figsize=(20, 7))
        for func, data in self.files_data_including_failed.items():
            y = np.array(data)
            x = np.array(range(size_of_data_set))
            cubic_interploation_model = interp1d(x, y, kind="cubic")
            X_Y_Spline = make_interp_spline(x, y)

            # Returns evenly spaced numbers
            # over a specified interval.
            X_ = np.linspace(x.min(), x.max() , 500)
            Y_ = X_Y_Spline(X_)
            # Plotting the Graph
            plt.plot(X_, Y_, label=func)

        plt.ylim(0, 5)
        plt.xlabel("sample index")
        plt.ylabel("Perturbation Norm")
        plt.title("Perturbation intensity on each sample")
        plt.legend()
        plt.show()

    def create_box_plot(self):
        plt.figure(figsize=(10, 7))
        plt.grid(visible=True, which='both')
        data = [norm for norm in self.files_data.values() if norm != 0]
        funcs = self.files_data.keys()
        plt.ylabel("Perturbation Norm")
        plt.title("Distribution of perturbation's Norm")
        plt.boxplot(data, labels=funcs)
        plt.show()

    def create_percentage_bar(self):
        funcs = self.percentage_of_failures.keys()
        failed_percent = self.percentage_of_failures.values()
        plt.bar(funcs, failed_percent)
        plt.ylabel("Percentage")
        plt.title("Percentage of failed attack attempts by HopSkipJump")
        plt.show()

    def network_accuracy_bar(self):
        X = accuracies_after.keys()
        before = accuracies_before.values()
        after = accuracies_after.values()

        X_axis = np.arange(len(X))

        plt.bar(X_axis - 0.2, before, 0.4, label='Before')
        plt.bar(X_axis + 0.2, after, 0.4, label='After')

        plt.xticks(X_axis, X)
        plt.ylabel("Test Accuracy")
        plt.title("DNN test accuracy before and after attacks")
        plt.grid(visible=True,which="both")
        plt.legend()
        plt.show()

    def read_results(self, path):
        with open(path, 'r') as file:
            file_data = []  # lists of perturbations, the indexes of the list are the same indexes of the image
            file_data_with_failure = []
            failure_count = 0
            file.readline()  # skip the header
            line = file.readline()
            while line:
                norm = line.split()[2]
                if norm != 'Attack':  # if attack failed
                    file_data.append(float(norm))
                    file_data_with_failure.append(float(norm))
                else:
                    failure_count += 1
                    file_data_with_failure.append(float(0))
                line = file.readline()
            # extract function name from file path:
            func = re.search(r'([a-z]+)_results_log\.txt', path).group(1)
            self.files_data[func] = file_data
            self.files_data_including_failed[func] = file_data_with_failure
            self.percentage_of_failures[func] = failure_count / size_of_data_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, action='store', help="Specify one file for processing"
                                                                  "or a directory for processing all *_results_log.txt")
    arguments = parser.parse_args()
    processor = GraphGenerator(arguments)
    processor.run()
