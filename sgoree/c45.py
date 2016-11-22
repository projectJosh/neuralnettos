# c45.py
# creates a set of rules according to the c4.5 algorithm

from id3 import split_on_attribute, gain
from collections import OrderedDict
from decision_rule import *
from decision_tree import *
import copy
import math

epsilon = 10e-6

DEBUG = False

# dataset is our dataset, attributes are tuples with a name and whether it is continuous or not, 
# attributes doesn't contain label, just attributes we're allowed to split on here

# contains functionality to return a tree rather than a rule list, this is not recommended for actual classification
# and is mostly included for visualization purposes
def train(dataset, attributes, label_attribute='label', split_info=True, return_tree=False):
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
		if return_tree: return LeafNode(prev_label)
		else: return [DecisionRule(OrderedDict(), prev_label, OrderedDict([(a,1) for a in attributes]))]
	# else find the attribute with the largest information gain
	max_gain = 0
	a_star = None
	threshold_star = -1
	is_continuous = False
	fractions = OrderedDict({})
	for attribute,continuous in attributes:
		if DEBUG: print("Considering ", attribute)
		if not continuous:
			# Deal with missing values of that attribute according to the totals
			totals = OrderedDict({})
			total = 0
			# count totals
			for instance in dataset:
				if instance[attribute] is not None:
					if instance[attribute] not in totals.keys(): totals[instance[attribute]] = 0
					# ',' is the multiplier on this instance
					totals[instance[attribute]] += instance[',']
					total += instance[',']
			dataset_no_missing_values = []
			for instance in dataset:
				# if the attribute is missing
				if instance[attribute] is None:
					for v, t in totals.items():
						# split the instance for each observed value of the attribute
						new_instance = copy.deepcopy(instance)
						new_instance[attribute] = v
						# multiply each copy's weight (,) by the fraction of the dataset with that attribute
						new_instance[','] = instance[','] * t/total
						dataset_no_missing_values.append(new_instance)
				else: dataset_no_missing_values.append(instance)
			g = gain(dataset, lambda l: split_on_attribute(l,attribute))/(split_information(dataset, attribute) if split_info else 1)
			if g > max_gain:
				a_star = attribute
				max_gain = g
				is_continuous = False
				fractions = OrderedDict([(v,t/total) for v,t in totals.items()])
		# Deal with splitting on continuous attributes - each split is binary, but we leave that attribute in the tree 
		#(hopefully its gain is really low now unless it's still helpful)
		else:
			dataset_no_missing_values = []
			for instance in dataset:
				if instance[attribute] is not None:
					dataset_no_missing_values.append(instance)
			sorted_list_of_values = sorted([(instance[attribute], instance[label_attribute], instance[',']) 
				for instance in dataset_no_missing_values], key=lambda v: float(v[0]))
			for i in range(len(sorted_list_of_values)):
				# detect a border between labels
				if sorted_list_of_values[i][1] != sorted_list_of_values[i-1][1]:
					threshold = float(sorted_list_of_values[i][0])
					g = gain(dataset_no_missing_values, lambda l:split_on_continuous_attribute(l, attribute, threshold)
						)/(split_information(dataset_no_missing_values, attribute) if split_info else 1)
					if g > max_gain:
						a_star = attribute
						threshold_star = threshold
						max_gain = g
						is_continuous = True
						# what an awful line of code
						fractions = OrderedDict([(-1, sum(float(val[2]) for val in sorted_list_of_values[:i])/sum(float(val[2]) 
							for val in sorted_list_of_values)),
						(1, sum(float(val[2]) for val in sorted_list_of_values[i:])/sum(float(val[2]) 
							for val in sorted_list_of_values))])

	if a_star is None:
		counts = OrderedDict()
		for label, subdataset in split_on_attribute(dataset, label_attribute).items():
			counts[label] = sum([instance[','] for instance in subdataset])
		total = sum(counts.values())
		majority = max(counts)
		if majority is None:
			print("Error: Trying to create a tree without training data")
			return None
		if return_tree: return LeafNode(majority)
		else: return [DecisionRule(OrderedDict(), majority, OrderedDict(), certainty = counts[majority]/total)]
	# split on a* and recurse
	if DEBUG: print("A_star!", a_star)
	final_rules = []
	if is_continuous:
		unknowns_lower = []
		unknowns_higher = []
		dataset_no_missing_values = []
		for instance in dataset:
			if instance[a_star] is None:
				new_instance_lower = copy.deepcopy(instance)
				new_instance_lower[',']*= fractions[-1]
				unknowns_lower.append(new_instance_lower)
				new_instance_upper = copy.deepcopy(instance)
				new_instance_upper[',']*= fractions[1]
				unknowns_higher.append(new_instance_upper)
			else: dataset_no_missing_values.append(instance)
		split_dataset = split_on_continuous_attribute(dataset_no_missing_values, a_star, threshold_star)

		if return_tree:
			left_subtree = train(split_dataset[-1] + unknowns_lower, attributes, label_attribute, return_tree=True)
			right_subtree = train(split_dataset[1] + unknowns_higher, attributes, label_attribute, return_tree=True)
			values_to_subtrees = OrderedDict([((threshold_star, -1), left_subtree), ((threshold_star, 1), right_subtree)])
			final_rules = DecisionNode(a_star, values_to_subtrees)
		else:
			lower_rules = train(split_dataset[-1] + unknowns_lower, attributes, label_attribute)
			upper_rules = train(split_dataset[1] + unknowns_higher, attributes, label_attribute)
			values_to_rules = OrderedDict([((threshold_star,-1),(lower_rules,fractions[-1])), ((threshold_star,1),(upper_rules, fractions[1]))])
			final_rules = build_decisionrules(a_star, values_to_rules)
	else:
		split_dataset = split_on_attribute(dataset, a_star)
		remaining_attributes = copy.deepcopy(attributes)
		for i in range(len(remaining_attributes)):
			if remaining_attributes[i][0] == a_star:
				del(remaining_attributes[i])
				break
		if return_tree:
			values_to_subtrees = OrderedDict()
			for value,subdataset in split_dataset.items():
				subtree = train(subdataset, remaining_attributes, label_attribute, return_tree=True)
				values_to_subtrees[value] = subtree
			final_rules = DecisionNode(a_star, values_to_subtrees)
		else:
			values_to_rules = OrderedDict()
			for value,subdataset in split_dataset.items():
				rules = train(subdataset, remaining_attributes, label_attribute)
				values_to_rules[(value,0)] = (rules, fractions[value])
			final_rules = build_decisionrules(a_star, values_to_rules)
	if DEBUG: 
		print("\nSplitting on", a_star)
		for rule in final_rules:
			print(rule)
	return final_rules



def prune(decision_rules, validation_set, label_attribute='label'):
	if DEBUG: print("\nStart Pruning\n")
	for i in range(len(decision_rules)):
		if DEBUG: print("Rule ", str(i), "of", len(decision_rules))
		changed = True
		# keep track of precision
		start_precision = precision(lambda inst: decision_rules[i].classify(inst)[0], validation_set, label_attribute)
		if DEBUG: print("Start Precision: ", start_precision)
		temp_rule = None
		while changed:
			changed=False
			for j,(attribute,precondition) in enumerate(decision_rules[i].preconditions.items()):
				# remove precondition
				temp_rule = copy.deepcopy(decision_rules[i])
				temp_rule.preconditions.pop(attribute)
				new_precision = precision(lambda inst: temp_rule.classify(inst)[0], validation_set)
				if new_precision > start_precision and new_precision != 0:
					changed=True
					temp_rule.certainty = new_precision
					if DEBUG: print("Changed ", decision_rules[i], "with precision", start_precision, "to", temp_rule, "with precision", new_precision)
					if DEBUG: print(temp_rule)
					start_precision = new_precision
					break
			if changed: decision_rules[i] = temp_rule
				# if precision is not higher than the default
				# add precondition to the new precondition list
		if DEBUG: print("Final Precision: ", start_precision)
	return decision_rules

# classifier should be a function that returns a classification if the preconditions apply, else None
def precision(classifier, dataset, label_attribute='label'):
	correct = 0
	incorrect = 0
	for instance in dataset:
		if classifier(instance) is not None and classifier(instance) == instance[label_attribute]:
			correct+=1
		elif classifier(instance) is not None and classifier(instance) != instance[label_attribute]:
			incorrect+=1
	return correct/(correct+incorrect) if correct + incorrect > 0 else 0


def split_information(dataset, attribute, label_attribute='label'):
	counts = OrderedDict()
	total = 0
	for instance in dataset:
		if instance[attribute] not in counts.keys():
			counts[instance[attribute]] = instance[',']
		else:
			counts[instance[attribute]] += instance[',']
		total += instance[',']
	total_entropy = 0
	for val, count in counts.items():
		total_entropy += (count/total * math.log2(count/total))
	# avoid divide by zero errors: split_information cannot be zero
	if total_entropy == 0:
		total_entropy += epsilon
	return -total_entropy


# threshold should be a number, returned dictionary[1] should be the list of values greater or equal, -1 should be the list of values less
def split_on_continuous_attribute(dataset, attribute, threshold):
	split_dataset = OrderedDict({1:[], -1:[]})
	for d in dataset:
		if float(d[attribute]) >= threshold:
			split_dataset[1].append(d)
		else:
			split_dataset[-1].append(d)
	return split_dataset

def c45(attributes, dataset, label_attribute='label', pruning=True, split_info=True, return_tree=False):
	training_set = dataset[:math.floor(2*len(dataset)/3)]
	validation_set = dataset[math.floor(2*len(dataset)/3):]
	attributes.remove(label_attribute)
	# check whether each attribute is continuous and set the multiplier of each instance
	fancy_attributes = []
	for attribute in attributes:
		values = []
		for instance in training_set:
			instance[','] = 1.0 # multiplier - nothing will ever have the name ','
			contains = False
			for val in values:
				if val == instance[attribute]:
					contains = True
					break
			if not contains:
				values += [instance[attribute]]
		# more than 20 probs means continuous
		if len(values) > 20: 
			if DEBUG: print(attribute, "considered continuous with ", len(values), "values")
			fancy_attributes.append((attribute, True))
		else: fancy_attributes.append((attribute, False))

	if return_tree:
		return train(training_set, fancy_attributes, label_attribute=label_attribute, split_info=split_info, return_tree=True)
	else:
		decision_rule_list = train(training_set, fancy_attributes, label_attribute=label_attribute, split_info=split_info)
		if pruning: pruned_rule_list = prune(decision_rule_list, training_set + validation_set, label_attribute)
		else: pruned_rule_list = decision_rule_list
		sorted_rule_list = sorted(pruned_rule_list, key=lambda r: r.certainty)
		return sorted_rule_list #prune(pruned_rule_list, validation_set, label_attribute)