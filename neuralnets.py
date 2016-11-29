#Josh Parker and Sam Goree's Neural Nets homework #3 
from sklearn.neural_network import MLPClassifier
import encoder
import numpy
import copy

#So, if i understand correctly, we're going to have a main class, an IO class, and this neural nets class?
#We run the main class, which uses the IO class to take a file path and create an array of instances, and then we send them through the neural nets class stuff,
# and then use the results and the IO class to output a file.

#replicate a main class, use iotools.py to create the instances, then use scikitLearn in neuralNets as the algorithm class.
class neural_nets:
    
    def train(self, trainingSet, attributes, hidden_layer_sizes, num_training_iterations, seed, onehot=False):
        #In order to fit a training set, we need to pass in an array of
        #the encodings of the training set and the number of attributes?
        #As well as the list of all labels.
        encrypt = encoder.encoder()
        if onehot:
            encodings, vectors = encrypt.encode(trainingSet, attributes)
        else:
            encodings = []
            att = copy.deepcopy(attributes)
            att.pop('label')
            for instance in trainingSet:
                encodings.append([float(instance[a]) for a in att])
            _, vectors = encrypt.encode(trainingSet, attributes)
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver='sgd', batch_size=len(trainingSet), max_iter=num_training_iterations,
            random_state=1, momentum=0)
        #X is a 2D array of the encodings of the trainingSet, and the list of attributes.
        
        labels = []
        for i in range(len(vectors)):
            labels = numpy.append(labels, numpy.argmax(vectors[i]))
        self.mlp.fit(encodings, labels)
        #Y is a 1D array of the different labels for the trainingSet.
        
    def classify(self, testSet, attributes, labels, onehot=False):
        #In order to classify, we use predict given the test set.
        #X is a 2D array of the encodings of the testSet, and the list of attributes.
        decrypt = encoder.encoder()
        confusionMatrix = [[0]*len(labels) for _ in range(len(labels))] #This is a 2D array, with dimensions: number of labels x number of labels.
        results = [] #This array stores the predictions in encoding form.
        testEncodings, testLabels = decrypt.encode(testSet, attributes)
        if not onehot:
            testEncodings = []
            att = copy.deepcopy(attributes)
            att.pop('label')
            for instance in testSet:
                testEncodings.append([float(instance[a]) for a in att])
        results = self.mlp.predict(testEncodings)
        #Now, we start a counter. We use the counter as the index of both testSetEncodings and results.
        #Then, we go through and find the index of the 1 in each of them. Use those as the 
        #coordinates in confusionMatrix.
        counter = 0
        while counter < len(testEncodings):
            x = numpy.argmax(testLabels[counter])
            y = int(results[counter])
            confusionMatrix[x][y] = confusionMatrix[x][y] + 1
            counter = counter+1
        """file = open(filename,'w') - turns out I don't actually need the file stuff
        for y in confusionMatrix:
            for x in confusionMatrix[y]:
                file.write(confusionMatrix[x][y])
            file.write('\n')"""
        return confusionMatrix, self.mlp.score(testEncodings, numpy.argmax(testLabels, axis=1))
        
    #What do we need to return? How do we process it? How do we send the instances into this, for both training and predicting?