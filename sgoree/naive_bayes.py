# naive_bayes.py
# takes a dataset and calculates the probabilities that an attribute leads to a label using bayes' rule
# Designed to be compatable with id3 and c4.5

from collections import OrderedDict
from math import log
import sys

DEBUG = False


class BayesianClassifier:

	# dataset is a list of instance dictionaries, which go from attributes to values, label_attribute should not be in attributes
	def train(self, dataset, attributes, label_attribute='label'):
		self.label_attribute=label_attribute
		# we need to count the probability of each attribute value given a label value
		# (labelvalue ((attribute, value), frequency)) dict structure, corresponds to P(attributevalue|labelvalue)
		self.conditional_counts = OrderedDict()
		# ((attribute, value) frequency) dict structure for totals, corresponds to P(attributevalue)
		self.total_counts = OrderedDict()
		self.dataset_len = len(dataset)
		# should map from an attribute to a list of values
		self.num_values = OrderedDict()

		for instance in dataset:
			label = instance[label_attribute]
			# if a dictionary for this label value doesn't exist, create one
			if label not in self.conditional_counts.keys():
				self.conditional_counts[label] = OrderedDict()
				self.total_counts[(label_attribute, label)] = 1
			else:
				self.total_counts[(label_attribute, label)] += 1
			for attribute in attributes:
				if attribute not in instance.keys():
					instance[attribute] = 0
				value = instance[attribute]
				if attribute not in self.num_values.keys():
					self.num_values[attribute] = []
				if value not in self.num_values[attribute]:
					self.num_values[attribute].append(value)
				# if there is no element for attribute, value pair yet, create it
				if (attribute,value) not in self.conditional_counts[label].keys():
					self.conditional_counts[label][(attribute,value)] = 1
				else: self.conditional_counts[label][(attribute,value)] += 1
				# deal with total frequency
				if (label_attribute, label) not in self.total_counts.keys():
					self.total_counts[(label_attribute, label)] = 1
				else:
					self.total_counts[(label_attribute, label)] += 1


		# I'm pretty sure that's all there is to it, right?
		if DEBUG:
			print("conditional counts", self.conditional_counts)
			print("total counts", self.total_counts)

	# applies bayes' rule to an instance to get the probability that it has a given label
	# P(A|B1 u B2 u B3...) = P(B1|A) * P(B2|A) * ... * P(A)/(P(B1) * P(B2) * ...
	def classify(self, instance, attributes):
		p_label = OrderedDict()
		# loop through possible label values and find the probabilities
		for label_value in self.conditional_counts.keys():
			total_conditional_log_prob = 0 # log(1) = 0
			total_log_probability_of_attributes = 0
			# loop through attributes and take the sum of the logs of the counts
			for attribute in attributes:
				if attribute not in instance.keys():
					instance[attribute] = 0
				value = instance[attribute]
				# handle missing values
				if (attribute,value) not in self.conditional_counts[label_value].keys():
					self.conditional_counts[label_value][(attribute,value)] = 0
				if attribute not in self.num_values.keys():
					# wow we fucked up
					print("Unknown attribute in test set")
					sys.exit(1)
				if value not in self.num_values[attribute]:
					self.num_values[attribute].append(value) # I'm not going to recalculate things I saw before, this is fine
				if (attribute,value) not in self.total_counts.keys():
					self.total_counts[(attribute,value)] = 0


				# log(a) + log(b) = log(a * b)
				if DEBUG: print("prob of ", attribute, value, (self.conditional_counts[label_value][(attribute,value)] + 1 )/(self.total_counts[(self.label_attribute, label_value)] + len(self.num_values[attribute])))
				total_conditional_log_prob += log((self.conditional_counts[label_value][(attribute,value)] + 1)
					/(self.total_counts[(self.label_attribute, label_value)] + len(self.num_values[attribute])))

			log_p_label_given_attributes = total_conditional_log_prob + \
			log((self.total_counts[(self.label_attribute, label_value)])/self.dataset_len) 

			if DEBUG: print("Log probability of ", instance, "being labeled", label_value, "is", log_p_label_given_attributes)
			p_label[label_value] = log_p_label_given_attributes
		v=list(p_label.values())
		k=list(p_label.keys())
		if DEBUG: print("Max probability:", k[v.index(max(v))])
		return k[v.index(max(v))]