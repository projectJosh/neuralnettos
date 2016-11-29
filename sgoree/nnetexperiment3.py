# nnetexperiment3.py
# runs on optical digit for 100 training iterations with varied hidden layer sizes (5,10,20,50)

import id3
import naive_bayes
import iotools
from main import test
import decision_rule
from collections import OrderedDict
import sys
import math
import copy

sys.path.append('../')
import neuralnets

# should return accuracy as a percentage with the 95% confidence interval
def accuracy(confusion_matrix):
	num_correct= 0
	total = 0
	for i in range(len(confusion_matrix)):
		num_correct += confusion_matrix[i][i]
		total += sum(confusion_matrix[i])
	accuracy = num_correct/total
	variance = accuracy * (1-accuracy) / total
	stddev = math.sqrt(variance)
	confidence = 1.96 * stddev
	return accuracy, confidence

label_attribute = 'label'

attributes, full_dataset = iotools.parse_file('../data/opticalDigit.csv')
labels = []
for instance in full_dataset:
	if instance[label_attribute] not in labels: labels.append(instance[label_attribute])

fullList = OrderedDict()
for a in attributes:
	#Then loop through all possible values of that attribute, as an array, and set fullList[a] = that
	fullList[a] = []
	for instance in full_dataset:
		if instance[a] not in fullList[a]:
			fullList[a].append(instance[a])

hidden_sizes = [10,20,50,100]
num_iterations = 1000
accuracies = [[],[],[],[]]
for seed in range(100, 110):
	partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=4)
	training_set = []
	for p in partitions[:-1]:
		training_set += p
	test_set = partitions[-1]
	for a,h in zip(accuracies, hidden_sizes):
		print("NNets, Digits, seed =", seed, ', hidden_size =', h)
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, fullList, h, num_iterations, seed)
		nnet_matrix, _ = mlp.classify(test_set, fullList, labels)
		nnet_accuracy = accuracy(nnet_matrix)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(nnet_accuracy)
		a.append(nnet_accuracy[0])
for a,h in zip(accuracies, hidden_sizes):
	print("hidden size:", h)
	print("accuracy: ", str(sum(a)/len(a)))