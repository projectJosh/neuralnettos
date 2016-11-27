# nnetexperiment3.py
# runs on optical digit for 100 training iterations with varied hidden layer sizes (5,10,20,50)

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

attributes, full_dataset = iotools.parse_file('../DecisionTrees/data/opticalDigit.csv')
labels = []
for instance in full_dataset:
	if instance[label_attribute] not in labels: labels.append(instance[label_attribute])

hidden_sizes = [2,3,4,5]
accuracies = [[],[],[],[]]
for seed in range(100, 110):
	partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=0.7)
	training_set = []
	for p in partitions[:-1]:
		training_set += p
	test_set = partitions[-1]
	for a,h in zip(accuracies, hidden_sizes):
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, attributes)
		nnet_matrix = mlp.classify(test_set, labels)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(accuracy(nnet_matrix))
		a.append(accuracy(nnet_matrix))