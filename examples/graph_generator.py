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
    'sigmoid': 1.85,
    'gelu': 10.25,
    'elu': 10.36,
    'selu': 10.57,
    'tanh': 10.43,
}
accuracies_before = {
    'relu': 98.97,
    'sigmoid': 96.94,
    'gelu': 98.92,
    'elu': 98.74,
    'selu': 98.43,
    'tanh': 98.62,
}


class GraphGenerator:
    def __init__(self, args):
        self.files_data = {}
        self.files_counters = {}
        self.files_data_including_failed = {}
        self.files_counters_including_failed = {}
        self.percentage_of_failures = {}
        self.args = args
        self.inquiries_higher_than_relu = {}


    def run(self):
        if isfile(self.args.i):
            path = glob(self.args.i)[0]
            self.read_results(path)
        elif isdir(self.args.i):
            files = glob(self.args.i + "*_results_log.txt")
            for file in files:
                self.read_results(file)
        self.create_graphs_for_inquiries()
        self.percent_higher_than_relu()
        self.create_graphs()
        self.box_plot_perturbations()
        self.create_percentage_bar()
        self.network_accuracy_bar()
        self.box_plot_for_inquiries()


    def percent_higher_than_relu(self):
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(20, 14))
        funcs = self.files_counters_including_failed.keys()
        for func in funcs:
            if func != 'relu':
                self.inquiries_higher_than_relu[func] = 0
                for ind,count in enumerate(self.files_counters_including_failed[func]):
                    if count > self.files_counters_including_failed['relu'][ind]:
                        self.inquiries_higher_than_relu[func] += 1
                self.inquiries_higher_than_relu[func] /= (size_of_data_set/100);

        x = self.inquiries_higher_than_relu.keys()
        y = self.inquiries_higher_than_relu.values()
        plt.bar(x, y)
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.title("Percentage of attacks that had higher inquiries compared to ReLU")
        plt.show()

    def create_graphs_for_inquiries(self):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(20, 7))
        for func, data in self.files_counters_including_failed.items():
            y = np.array(data)
            x = np.array(range(size_of_data_set))
            #cubic_interploation_model = interp1d(x, y, kind="cubic")
            #X_Y_Spline = make_interp_spline(x, y)

            # Returns evenly spaced numbers
            # over a specified interval.
            #X_ = np.linspace(x.min(), x.max(), 500)
            #Y_ = X_Y_Spline(X_)
            # Plotting the Graph
            plt.plot(x, y, label=func)

        plt.ylim(24500, 24700)
        plt.xlabel("Sample index")
        plt.ylabel("number of inquiries")
        plt.title("Inquiries amount on each sample")
        plt.legend()
        plt.show()


    def create_graphs(self):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(20, 7))
        for func, data in self.files_data_including_failed.items():
            y = np.array(data)
            x = np.array(range(size_of_data_set))
            cubic_interploation_model = interp1d(x, y, kind="cubic")
            X_Y_Spline = make_interp_spline(x, y)

            # Returns evenly spaced numbers
            # over a specified interval.
            X_ = np.linspace(x.min(), x.max(), 500)
            Y_ = X_Y_Spline(X_)
            # Plotting the Graph
            plt.plot(X_, Y_, label=func)

        plt.ylim(0, 5)
        plt.xlabel("Sample index")
        plt.ylabel("Perturbation Norm")
        plt.title("Perturbation intensity on each sample")
        plt.legend()
        plt.show()

    def box_plot_perturbations(self):
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(20, 7))
        plt.grid(visible=True, which='both')
        data = [norm for norm in self.files_data.values() if norm != 0]
        funcs = self.files_data.keys()
        plt.ylabel("Perturbation Norm")
        plt.title("Distribution of perturbation's Norm")
        plt.boxplot(data, labels=funcs)
        plt.show()

    def box_plot_for_inquiries(self):
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(20, 7))
        plt.grid(visible=True, which='both')
        counters = self.files_counters.values()
        funcs = self.files_data.keys()
        plt.ylabel("number of inquiries")
        plt.title("Distribution of Number of Inquiries")
        plt.boxplot(counters,labels=funcs)
        plt.show()

    def create_percentage_bar(self):
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(20, 14))
        funcs = self.percentage_of_failures.keys()
        failed_percent = self.percentage_of_failures.values()
        # funcs = [func for func in funcs if func not in ['relu','sigmoid']]
        # failed_percent = [val for val in failed_percent if val > 0]
        plt.bar(funcs, failed_percent)
        plt.ylabel("Percentage")
        plt.ylim(0,40)
        plt.title("Percentage of failed attacks")
        plt.show()

    @staticmethod
    def network_accuracy_bar():
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(20, 14))
        X = accuracies_after.keys()
        before = accuracies_before.values()
        after = accuracies_after.values()

        X_axis = np.arange(len(X))

        plt.bar(X_axis - 0.2, before, 0.4, label='Before')
        plt.bar(X_axis + 0.2, after, 0.4, label='After')

        plt.xticks(X_axis, X)
        plt.ylabel("Test Accuracy")
        plt.title("DNN Accuracy on test dataset")
        plt.grid(visible=True, which="both")
        plt.legend(loc="right")
        plt.show()

    def read_results(self, path):
        with open(path, 'r') as file:
            file_data = []  # lists of perturbations, the indexes of the list are the same indexes of the image
            file_data_with_failure = []
            file_counters = []
            file_counters_with_failure = []
            failure_count = 0
            func = re.search(r'(\w+)_results_log\.txt', path).group(1)
            file.readline()  # skip the header
            line = file.readline()
            while line:
                acc_before = re.search(r'Test\s+accuracy:\s+(\d+\.\d+)', line)
                acc_after = re.search(r'Test\s+accuracy\s+on\s+adversarial\s+sample:\s+(\d+\.\d+)', line)
                if acc_before:
                    accuracies_before[func] = float(acc_before.group(1))
                    line = file.readline()
                    continue
                elif acc_after:
                    accuracies_after[func] = float(acc_after.group(1))
                    line = file.readline()
                    continue
                norm = line.split()[2]
                count = line.split()[1]
                if norm != 'Attack':  # if attack not failed
                    file_counters.append(int(count))
                    file_counters_with_failure.append(int(count))
                    file_data.append(float(norm))
                    file_data_with_failure.append(float(norm))
                else:
                    failure_count += 1
                    file_data_with_failure.append(float(0))
                    file_counters_with_failure.append(int(count))
                line = file.readline()
            # extract function name from file path:
            self.files_counters[func] = file_counters
            self.files_counters_including_failed[func] = file_counters_with_failure
            self.files_data[func] = file_data
            self.files_data_including_failed[func] = file_data_with_failure
            self.percentage_of_failures[func] = (failure_count*100) / size_of_data_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, action='store', help="Specify one file for processing"
                                                                  "or a directory for processing all *_results_log.txt")
    arguments = parser.parse_args()
    processor = GraphGenerator(arguments)
    processor.run()
