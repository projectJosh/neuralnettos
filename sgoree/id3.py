# id3.py
# builds a decision tree according to the ID3 algorithm from a training set

from decision_tree import *
from iotools import *

import copy
import math
from collections import OrderedDict

def train(dataset, attributes, label_attribute='label'):
	# if all the instances have the same label, return a leaf node
	homogeneous = True
	prev_label = None
	for d in dataset:
		if d[label_attribute] != prev_label and prev_label is not None:
			homogeneous = False
			break
		elif prev_label is None:
			prev_label = d[label_attribute]
	if homogeneous:
		return LeafNode(dataset[0][label_attribute])
	# else find the attribute with the largest information gain
	max_gain = 0
	a_star = None
	for attribute in attributes:
		g = gain(dataset, lambda l: split_on_attribute(l,attribute))
		if g > max_gain:
			a_star = attribute
			max_gain = g
	if a_star is None:
		majority = None
		count = 0
		for k,v in split_on_attribute(dataset, label_attribute).items():
			if len(v) > count:
				majority = k
				count = len(v)
		if majority is None:
			print("Error: Trying to create a tree without training data")
			return None
		return LeafNode(majority)
	# split on a*
	split_dataset = split_on_attribute(dataset, a_star)
	# create a OrderedDictionary mapping values of a* to subtrees, created recursively
	subtrees_OrderedDict = OrderedDict()
	remaining_attributes = copy.deepcopy(attributes)
	remaining_attributes.remove(a_star)
	for k,v in split_dataset.items():
		subtrees_OrderedDict[k] = train(v, remaining_attributes, label_attribute)
	r = DecisionNode(a_star, subtrees_OrderedDict)
	return r

# returns a OrderedDictionary mapping values of the attribute to sub-datasets
def split_on_attribute(dataset, attribute):
	split_dataset = OrderedDict()
	for d in dataset:
		if not d[attribute] in split_dataset.keys():
			split_dataset[d[attribute]] = []
		split_dataset[d[attribute]].append(d)
	return split_dataset

def entropy(dataset, label_attribute='label'):
	total = sum([dataset[i][','] for i in range(len(dataset))])
	split_dataset = split_on_attribute(dataset, label_attribute)
	s = 0
	for partition in split_dataset.values():
		partition_len = sum([partition[i][','] for i in range(len(partition))])
		s += (partition_len/total) * math.log2(partition_len/total)
	return -s


# return the information gain if the given dataset is split on the given attribute
def gain(dataset, splitter, label_attribute='label'):
	split_dataset = splitter(dataset)
	total = 0
	for partition in split_dataset.values():
		total += (len(partition)/len(dataset) * entropy(partition))
	return entropy(dataset) - total



def id3(attributes, dataset, label_attribute='label'):
	attributes.remove(label_attribute)
	for instance in dataset:
		instance[','] = 1.0 # multiplier - nothing will ever have the name ','
	tree = train(dataset, attributes, label_attribute=label_attribute)
	return tree