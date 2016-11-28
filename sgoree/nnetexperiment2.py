# experiment2.py
# run on opticalDigit first four times with different training/test set proportions for a fixed number of training iterations (1000)
# then runs on different numbers of training iterations with training/test set proportion of 0.7

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

# first part - vary proportions
proportions = [2,3,4,5]
accuracies_a = [[],[],[],[]]
for p,a in zip(proportions, accuracies_a):
	for seed in range(100, 110):
		# split the dataset according to that proportion
		partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=p)
		training_set = []
		for p in partitions[:-1]:
			training_set += p
		test_set = partitions[-1]
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, attributes)
		nnet_accuracy, nnet_matrix = mlp.classify(test_set, attributes, labels)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(nnet_accuracy)
		a.append(nnet_accuracy)
	print("Proportion:1/" + str(p))
	print("accuracy: ", str(sum(a)/len(a)))

# second part - vary training iterations
accuracies_b = [[] for i in range(10)]
for seed in range(100, 110):
	partitions = iotools.split_dataset(full_dataset, seed=seed, num_partitions=0.7)
	training_set = []
	for p in partitions[:-1]:
		training_set += p
	test_set = partitions[-1]
	for a,num_iterations in zip(accuracies_b, range(0, 2000, 200)):
		mlp = neuralnets.neural_nets()
		mlp.train(training_set, attributes)
		nnet_accuracy, nnet_matrix = mlp.classify(test_set, attributes, labels)
		print(iotools.print_confusion_matrix(nnet_matrix, labels))
		print(nnet_accuracy)
		a.append(nnet_accuracy)

