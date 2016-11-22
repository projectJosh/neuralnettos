# book_preprocessing.py
# functions to preprocess book data and test samples in the style of the file from blackboard

from StemmingUtil import *
from collections import Counter,OrderedDict
import sys

DEBUG = False

# returns word stem frequencies for every word encountered in the file as an OrderedDictionary
def parse_file(path, stop_words):
	file = open(path)
	label = file.readline()
	if label[1] == '#':
		if DEBUG: print('label found for ' + path + ' was ' + label[2:-1])
		label = label[2:-1]
	else:
		print("No label found in file " + path)
		sys.exit(1)
	tokens = parseTokens(file.read())
	# remove stop words
	for word in tokens:
		if word in stop_words:
			tokens.remove(word)
	# stem the remaining tokens
	stems = createStems(tokens)
	frequencies = OrderedDict(Counter(stems))
	if DEBUG: print(frequencies)
	frequencies['label*'] = label
	return frequencies

# parse a file in the style of test_set
# #######################
# author
# sample
def parse_test_instances(path):
	state = 0
	instances = []
	file = open(path)
	for line in file:
		print(line)
		if line[0] == '#':
			state +=1
			continue
		elif state == 1:
			instances.append(OrderedDict())
			instances[-1]['label*'] = str(line)[:-1]
			state += 1
		elif state == 2:
			frequencies = OrderedDict(Counter(createStems(parseTokens(str(line)))))
			instances[-1] = {**instances[-1], **frequencies}
			state = 0
	return instances