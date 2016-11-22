# iotools.py
# file io tools for decision trees

import random
import os
import math
from collections import OrderedDict

# parse a data file, returns a list of attributes and a list of OrderedDictionaries,
# one per instance, which map attributes to values. Important: one attribute must be 'label'
def parse_file(path):
	data_file = open(path).read().split('\n')
	attributes = data_file[0].replace('"','').split(',')
	raw_data = data_file[1:]
	data = []
	for line in raw_data[:-1]:
		values = line.replace('"','').split(',')
		values = [None if value == '?' else value for value in values]
		d = OrderedDict((a,v) for a,v in zip(attributes,values))
		data += [d]
	return attributes,data

# split a dataset stochastically into num_partitions equal sections
# if the seed isn't specified, it'll be the current system time
def split_dataset(data, seed=None, num_partitions=3):
	partitions = []
	random.seed(seed)
	for i in range(num_partitions): partitions.append([])
	for i in range(len(data)):
		n = math.floor(random.random() * num_partitions)
		partitions[n] += [data[i]]
	return partitions

# output a confusion matrix for the 
def output_confusion_matrix(matrix, labels, dataset_name, algorithm_name, seed, output_dir='.'):
	# open the file
	if not os.path.isdir(output_dir): os.mkdir(output_dir)
	output_file = open(output_dir + 'results_' + dataset_name + '_' + algorithm_name + '_' + str(seed) + '.csv', 'w')
	output_file.write(print_confusion_matrix(matrix,labels))

def print_confusion_matrix(matrix, labels):
	retval = ''
	# write labels at the top
	for l in labels:
		retval += str(l) + ','
	retval += '\n'
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			retval += str(matrix[i][j]) + ','
		retval += str(labels[i]) + '\n'
	return retval

def output_graph_image_source(tree, path):
	f = open(path, 'w')
	_, text = graph_helper(tree)
	f.write('digraph G {\n' + text + '}')

# takes a tree, returns a name for the tree's root and a string for the subtree's diagram
def graph_helper(tree):
	retval = ''
	if tree.isleaf:
		return tree.label + str(hash(tree) % 1000), tree.label + str(hash(tree) % 1000) + '[label = "' + tree.label + '"];\n'
	else:
		for value,subtree in tree.subtrees.items():
			name, subtree_str = graph_helper(subtree)
			retval += tree.attribute+ str(hash(tree) % 1000) + '[label = "' + tree.attribute + '"];\n'
			retval += tree.attribute + str(hash(tree) % 1000) + '-> ' + name + ' [label = " ' + str(value) + '"];\n'
			retval += subtree_str
		return tree.attribute + str(hash(tree) % 1000), retval