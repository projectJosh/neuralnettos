# decision_tree.py
# data structures for decision trees, for now just id3 compatable
import random

DEBUG = False

class TreeNode:
	def __init__(self, isleaf):
		self.isleaf = isleaf

	# abstract method
	def classify(self, instance):
		raise NotImplementedError

	def rec_str(self, depth):
		raise NotImplementedError

	def __str__(self):
		return self.rec_str(0)

# leaf node class
class LeafNode(TreeNode):
	def __init__(self, label):
		super(LeafNode, self).__init__(True)
		self.label = label
	def classify(self, instance):
		return self.label

	def rec_str(self, depth):
		retval = ''
		for i in range(depth): retval += '    '
		retval += self.label + '\n'
		return retval

# TODO: Store partition amounts in the tree
class DecisionNode(TreeNode):

	# constructor takes the string attribute to classify on 
	# and a dictionary mapping values of the attribute to Tree objects
	def __init__(self, attribute, values_to_subtrees):
		super(DecisionNode, self).__init__(False)
		self.attribute = attribute
		self.subtrees = values_to_subtrees

	# instance should be a dict mapping attributes to values
	def classify(self, instance):
		if instance[self.attribute] not in self.subtrees.keys():
			if DEBUG: print("Unable to classify", instance)
			return list(self.subtrees.values())[random.randint(0,len(self.subtrees)-1)].classify(instance)
		return self.subtrees[instance[self.attribute]].classify(instance)

	def rec_str(self, depth):
		retval = ''
		indent = ''
		for i in range(depth): indent += '    '
		retval += indent + self.attribute + '\n'
		for k,v in self.subtrees.items():
			retval += indent + '  ' + k + '\n'
			retval += v.rec_str(depth+1)
		return retval + '\n'
		