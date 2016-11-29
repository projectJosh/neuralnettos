# nnetexperiment1.py
# runs id3, nnets, NB on two data sets 30 times each

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

num_hidden_neurons = 20
num_training_iterations = 1000

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

monks_attributes, monks_full_dataset = iotools.parse_file('../data/monks1.csv')
monks_labels = []
label_attribute='label'
for instance in monks_full_dataset:
	if instance[label_attribute] not in monks_labels: monks_labels.append(instance[label_attribute])

#We need instead of nb_attributes, to pass in an ordered dictionary, whose keys are the FULL list of attributes,
#and whose values are the FULL list of values, from all instances.
monks_fullList = OrderedDict()
for a in monks_attributes:
	#Then loop through all possible values of that attribute, as an array, and set fullList[a] = that
	monks_fullList[a] = []
	for instance in monks_full_dataset:
		if instance[a] not in monks_fullList[a]:
			monks_fullList[a].append(instance[a])

digits_attributes, digits_full_dataset = iotools.parse_file('../data/opticalDigit.csv')
digits_labels = []
for instance in digits_full_dataset:
	if instance[label_attribute] not in digits_labels: digits_labels.append(instance[label_attribute])

#We need instead of nb_attributes, to pass in an ordered dictionary, whose keys are the FULL list of attributes,
#and whose values are the FULL list of values, from all instances.
digits_fullList = OrderedDict()
for a in digits_attributes:
	#Then loop through all possible values of that attribute, as an array, and set fullList[a] = that
	digits_fullList[a] = []
	for instance in digits_full_dataset:
		if instance[a] not in digits_fullList[a]:
			digits_fullList[a].append(instance[a])

results = [[[],[],[]],[[],[],[]]]
for seed in range(100,130):
	partitions = iotools.split_dataset(monks_full_dataset, seed=seed, num_partitions=3)
	monks_training_set = []
	for p in partitions[:-1]:
		monks_training_set += p
	monks_test_set = partitions[-1]
	# run id3 on monks
	tree = id3.id3(copy.deepcopy(monks_attributes), monks_training_set)
	id3_labels, id3_matrix = test(monks_test_set, tree.classify, copy.deepcopy(monks_labels))
	print("ID3, Monks, seed =", seed)
	print(iotools.print_confusion_matrix(id3_matrix, id3_labels))
	print(accuracy(id3_matrix))
	results[0][0].append(accuracy(id3_matrix)[0])
	# run neural nets on monks
	mlp = neuralnets.neural_nets()
	mlp.train(monks_training_set, monks_fullList, num_hidden_neurons, num_training_iterations, seed, True)
	nnet_matrix, nnet_accuracy = mlp.classify(monks_test_set, monks_fullList, monks_labels, True)
	nnet_accuracy = accuracy(nnet_matrix)
	print("NNet, Monks, seed =", seed)
	print(iotools.print_confusion_matrix(nnet_matrix, monks_labels))
	print(nnet_accuracy)
	results[0][1].append(nnet_accuracy[0])
	# run NB on monks
	nb_attributes = copy.deepcopy(monks_attributes)
	nb_attributes.remove(label_attribute) #Do we definitely want to do this??
	nb = naive_bayes.BayesianClassifier()
	nb.train(monks_training_set, nb_attributes)
	nb_labels, nb_matrix = test(monks_test_set, lambda inst: nb.classify(inst, nb_attributes), copy.deepcopy(monks_labels))
	print("NB, Monks, seed =", seed)
	print(iotools.print_confusion_matrix(nb_matrix, nb_labels ))
	print(accuracy(nb_matrix))
	results[0][2].append(accuracy(nb_matrix)[0])
	

	partitions = iotools.split_dataset(digits_full_dataset, seed=seed, num_partitions=3)
	digits_training_set = []
	for p in partitions[:-1]:
		digits_training_set += p
	digits_test_set = partitions[-1]
	# run id3 on digits
	tree = id3.id3(copy.deepcopy(digits_attributes), digits_training_set)
	id3_labels, id3_matrix = test(digits_test_set, tree.classify, digits_labels)
	print("ID3, Digits, seed =", seed)
	print(iotools.print_confusion_matrix(id3_matrix, id3_labels))
	print(accuracy(id3_matrix))
	results[1][0].append(accuracy(id3_matrix)[0])
	# run neural nets on digits
	mlp = neuralnets.neural_nets()
	mlp.train(digits_training_set, digits_fullList, num_hidden_neurons, num_training_iterations, seed)
	nnet_matrix, nnet_accuracy = mlp.classify(digits_test_set, digits_fullList, digits_labels)
	nnet_accuracy = accuracy(nnet_matrix)
	print("NNet, Digits, seed =", seed)
	print(iotools.print_confusion_matrix(nnet_matrix, digits_labels))
	print(nnet_accuracy)
	results[1][1].append(nnet_accuracy[0])
	# run NB on digits
	nb_attributes = copy.deepcopy(digits_attributes)
	nb_attributes.remove(label_attribute)
	nb = naive_bayes.BayesianClassifier()
	nb.train(digits_training_set, nb_attributes)
	nb_labels, nb_matrix = test(digits_test_set, lambda inst: nb.classify(inst, nb_attributes), digits_labels)
	print("NB, Digits, seed =", seed)
	print(iotools.print_confusion_matrix(nb_matrix, nb_labels))
	print(accuracy(nb_matrix))
	results[1][2].append(accuracy(nb_matrix)[0])


print("Totals")

print('ID3 monks', sum(results[0][0])/len(results[0][0]))
print('NNet monks', sum(results[0][1])/len(results[0][1]))
print('NB monks', sum(results[0][2])/len(results[0][2]))

print('ID3 digits', sum(results[1][0])/len(results[1][0]))
print('NNet digits', sum(results[1][1])/len(results[1][1]))
print('NB digits', sum(results[1][2])/len(results[1][2]))