# decision_rule.py
# reduction of a tree, instead of being a recursive structure is a single evaluation built recursively

import decision_tree
from collections import OrderedDict

class DecisionRule:

	# preconditions is a dictionary from attributes to (value, isthreshold), if isthreshold is 1, it is >, -1 <, 0 is not a threshold
	# ratios is a dict from attributes to probabilities
	# certainty is the fraction of instances with these preconditions that had the majority label
	def __init__(self, preconditions, label, ratios, certainty=1):
		self.preconditions = preconditions
		self.label = label
		self.probs = ratios
		self.certainty = certainty

	def classify(self, instance):
		prob = self.certainty
		for attribute, value in self.preconditions.items():
			# handle missing values
			if attribute not in instance.keys() or instance[attribute] == None:
				prob *= self.probs[attribute]
				continue
			if value[1] == 1 and float(instance[attribute]) < value[0]:
				return (None, 0)
			elif value[1] == -1 and float(instance[attribute]) >= value[0]:
				return (None, 0)
			elif value[1] == 0 and instance[attribute] != value[0]:
				return (None, 0)
		return (self.label, prob)

	def __str__(self):
		retval = 'IF '
		for attribute, value in self.preconditions.items():
			retval += attribute
			retval += ' < ' if value[1]==1 else (' > ' if value[1]==-1 else ' = ')
			retval += str(value[0]) + ' AND '
		return retval[:-5] + ' THEN ' + self.label + ' ' + str(self.certainty)

# values to rules should be a dictionary mapping (value, isthreshold) pairs to (decision rule, probability) pairs
def build_decisionrules(attribute, values_to_rules_and_probs):
	final_rules = []
	for value, rules_prob in values_to_rules_and_probs.items():
		for rule in rules_prob[0]:
			rule.preconditions[attribute] = value
			rule.probs[attribute] = rules_prob[1]
		final_rules += rules_prob[0]
	return final_rules

def classify_on_rule_list(instance, rule_list):
	for rule in rule_list:
		label, certainty = rule.classify(instance)
		if label is not None:
			return label
	return None