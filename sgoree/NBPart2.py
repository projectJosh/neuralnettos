# NBPart2.py
# contains code for the second handin of the naive bayes homework
# reads in a test set from sys.argv[1], trains on the books in directory sys.argv[2] with stop words from sys.argv[3] removed
# then classifies the test set and outputs a confusion matrix

# Note: each time this is run with a list of files, it pickles the result
# to load the pickle file, just use the argument '-p' after the test path

import sys
import os
import pickle
from math import log

from book_preprocessing import *
import naive_bayes
from main import test
from iotools import print_confusion_matrix

DEBUG=False


def main():
	test_path = sys.argv[1]
	p = False
	# handle pickle argument
	if len(sys.argv) > 2 and sys.argv[2] == '-p':
		p = True
		pickle_path = sys.argv[3]
	# else handle modularity for stop words and data locations
	else:
		if len(sys.argv) > 3:
			stop_words_path = sys.argv[3]
		else:
			stop_words_path = 'stop_words.txt'
		if len(sys.argv) > 2:
			training_path = sys.argv[2]
		else:
			training_path = 'data/'
	if not p:
		# parse stop words list
		stop_words = []
		for word in open(stop_words_path):
			if word[0] != '#':
				stop_words.append(word)

		training_set = []
		full_wordlist = []
		# parse training examples
		for f in os.listdir(training_path):
			instance = parse_file(training_path + f, stop_words)
			for word in instance.keys():
				if word not in full_wordlist and word != 'label*':
					full_wordlist.append(word)
			training_set.append(instance)
		# pickle it for faster loading in the future
		if DEBUG:
			print("Saving to pickle file temp.p...")
			pickle.dump((training_set, full_wordlist, stop_words), open('temp.p', 'wb'))
	else:
		print("Loading pickle file", pickle_path)
		training_set, full_wordlist, stop_words = pickle.load(open(pickle_path, 'rb'))


	# convert training set to probabilities
	training_set_probs = []
	for instance in training_set:
		training_set_probs.append(OrderedDict())
		total = 0
		for word in instance.keys():
			if word == 'label*': continue
			total += instance[word]
		for word in instance.keys():
			if word == 'label*': training_set_probs[-1][word] = instance[word]
			else: training_set_probs[-1][word] = instance[word] / total

	# P(label | instance) = prod P(word|label) for all words * (label count)/(len training set)
	def classifier(inst):
		log_probs = OrderedDict()
		for instance in training_set_probs:
			running_log_prob = 0
			for word in inst.keys():
				if word == 'label*': continue
				if word not in instance.keys():
					instance[word] = 1/len(full_wordlist)
				running_log_prob += log(instance[word]) * inst[word]
			log_probs[instance['label*']] = running_log_prob
			if DEBUG: print("prob of ", instance['label*'], " is ", running_log_prob)
		v=list(log_probs.values())
		k=list(log_probs.keys())
		if DEBUG: print("Max probability:", k[v.index(max(v))], "with log p:", max(v))
		return k[v.index(max(v))]



	# load test set
	test_set = parse_test_instances(test_path)
	if DEBUG: print(test_set)
	labels = []
	for instance in training_set + test_set:
		if instance['label*'] not in labels: labels.append(instance['label*'])

	labels, matrix = test(test_set, classifier, labels, label_attribute='label*')
	total_correct = 0
	for i in range(len(matrix)):
		total_correct += matrix[i][i]
	if DEBUG: 
		print("accuracy:", total_correct/len(test_set))
		print(matrix)
	open('results_textClassification.csv', 'w').write(print_confusion_matrix(matrix, labels))


if __name__ == '__main__':
	main()