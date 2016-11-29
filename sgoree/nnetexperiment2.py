# experiment2.py
# run on opticalDigit first four times with different training/test set proportions for a fixed number of training iterations (1000)
# then runs on different numbers of training iterations with training/test set proportion of 0.7

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

num_hidden_neurons = 20
num_training_iterations = 1000
# first part - vary proportions
proportions = [2,3,4,5]
accuracies_a = [[],[],[],[]]
for p,a in zip(proportions, accuracies_a):
	for seed in range(100, 110):
		print("NNets, Digits, seed =", seed, ', proportion = 1 /', p)
		# split the dataset according to that proportion
		partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=p)
		training_set = []
		for part in partitions[:-1]:
			training_set += part
		test_set = partitions[-1]
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, fullList, num_hidden_neurons, num_training_iterations, seed)
		nnet_matrix, _ = mlp.classify(test_set, fullList, labels)
		nnet_accuracy = accuracy(nnet_matrix)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(nnet_accuracy)
		a.append(nnet_accuracy[0])
	print("Proportion:1/" + str(p))
	print("accuracy: ", str(sum(a)/len(a)))

# second part - vary training iterations
accuracies_b = [[] for i in range(10)]
for seed in range(100, 110):
	partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=4)
	training_set = []
	for p in partitions[:-1]:
		training_set += p
	test_set = partitions[-1]
	for b,num_iterations in zip(accuracies_b, range(200, 2000, 200)):
		print("NNets, Digits, seed =", seed, ', num_iterations =', num_iterations)
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, fullList, num_hidden_neurons, num_iterations, seed)
		nnet_matrix, _ = mlp.classify(test_set, fullList, labels)
		nnet_accuracy = accuracy(nnet_matrix)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(nnet_accuracy)
		b.append(nnet_accuracy[0])
for b,num_iterations in zip(accuracies_b, range(200, 2000, 200)):
	print("num iterations: ", num_iterations)
	print("accuracy: ", str(sum(b)/len(b)))
