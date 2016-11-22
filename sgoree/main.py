# main.py
# handles command line arguments and calls decision tree algorithms
# Updated for naive bayes!


import iotools
import decision_tree
import id3
import c45
import decision_rule
import naive_bayes
import neuralnets

import sys
import itertools
import argparse
import re
from collections import OrderedDict

# takes a dataset (list of OrderedDicts, with labels) and a function and returns the confusion matrix
def test(dataset, classifier, labels, label_attribute='label'):
	# split the dataset based on its label attribute
	
	split_on_labels = id3.split_on_attribute(dataset, label_attribute)
	if None not in labels: labels.append(None)
	label_numbers = OrderedDict([(l,i) for i,l in enumerate(labels)])
	num_labels = len(label_numbers)
	confusion_matrix = []
	for label, number in label_numbers.items():
		subdataset = split_on_labels[label] if label in split_on_labels.keys() else []
		confusion_matrix += [[0] * num_labels]
		for instance in subdataset:
			l = classifier(instance)
			confusion_matrix[-1][label_numbers[l]] += 1
	return labels, confusion_matrix

#strings map to strings or floats, aka attribute category to values. Createa  function that takes a dataset, aka orderedDicts, and a list of attribute names, and
#converts each orderedDict entry to a 2D array. Result should be a 3D array, use numpy.

def main():
	if sys.argv[1] == '-h':
		print("TODO: use argparse and put pretty messages here")
		sys.exit(0)
	if(len(sys.argv) < 4):
		print("There's a missing parameter. Remember to include path to data, alg and seed")
		sys.exit(1)
	
	path = sys.argv[1]
	algorithm = sys.argv[2].lower()
	seed = sys.argv[3]
	if(len(sys.argv) == 5):
		output_dir = sys.argv[4]
	else: output_dir = './'
	attributes, full_dataset = iotools.parse_file(path)
	partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=3)
	labels = []
	label_attribute='label'
	for instance in full_dataset:
		if instance[label_attribute] not in labels: labels.append(instance[label_attribute])

	training_set = []
	for p in partitions[:-1]:
		training_set += p
	if algorithm == 'id3': 
		tree = id3.id3(attributes, training_set)
		#iotools.output_graph_image_source(tree, 'pretty_picture.gv')
		labels, matrix = test(partitions[-1], tree.classify, labels)
	elif algorithm == 'c4.5':
		rule_list = c45.c45(attributes, training_set)
		print("\nFinal Rules:\n")
		for rule in rule_list:
			print(rule)
		print(decision_rule.rule_list_to_tree(rule_list))
		labels, matrix = test(partitions[-1], lambda inst: decision_rule.classify_on_rule_list(inst, rule_list), labels)
	elif algorithm == 'c4.5np':
		rule_list = c45.c45(attributes, training_set, pruning=False)
		print("\nFinal Rules:\n")
		for rule in rule_list:
			print(rule)
		labels, matrix = test(partitions[-1], lambda inst: decision_rule.classify_on_rule_list(inst, rule_list), labels)
	elif algorithm == 'c4.5nsi':
		rule_list = c45.c45(attributes, training_set, split_info=False)
		print("\nFinal Rules:\n")
		for rule in rule_list:
			print(rule)
		labels, matrix = test(partitions[-1], lambda inst: decision_rule.classify_on_rule_list(inst, rule_list), labels)
	elif algorithm == 'naivebayes':
		attributes.remove(label_attribute)
		nb = naive_bayes.BayesianClassifier()
		nb.train(training_set, attributes)
		labels, matrix = test(partitions[-1], lambda inst: nb.classify(inst, attributes), labels)
	elif algorithm == 'neuralnets':
	        
	        nn = ?
	        nn.train(training_set, attributes)
	else:
		print("Sorry, that algorithm is not implemented yet")
		sys.exit(1)
	iotools.output_confusion_matrix(matrix, labels, re.sub(r'.*/([^/\.]*)\.csv', r'\1', path), algorithm, seed, output_dir)

if __name__ == '__main__':
	main()