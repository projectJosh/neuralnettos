#Josh Parker and Sam Goree's Neural Nets homework #3 
from sklearn.neural_network import MLPClassifier
import encoder
import numpy

#So, if i understand correctly, we're going to have a main class, an IO class, and this neural nets class?
#We run the main class, which uses the IO class to take a file path and create an array of instances, and then we send them through the neural nets class stuff,
# and then use the results and the IO class to output a file.

#replicate a main class, use iotools.py to create the instances, then use scikitLearn in neuralNets as the algorithm class.
class neural_nets:
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    
    def train(self, trainingSet, attributes):
        #In order to fit a training set, we need to pass in an array of
        #the encodings of the training set and the number of attributes?
        #As well as the list of all labels.
        encrypt = encoder
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        #X is a 2D array of the encodings of the trainingSet, and the list of attributes.
        encodings, vectors = encrypt.encode(encrypt, trainingSet, attributes)
        labels = []
        for i in vectors:
            numpy.append(labels, numpy.argmax(vectors[i]))
        self.mlp = clf.fit(encodings, labels)
        #Y is a 1D array of the different labels for the trainingSet.
        return self.mlp
        
    def classify(self, testSet, attributes, labels, filename):
        #In order to classify, we use predict given the test set.
        #X is a 2D array of the encodings of the testSet, and the list of attributes.
        decrypt = encoder
        confusionMatrix = [[0]*len(labels) for _ in range(len(labels))] #This is a 2D array, with dimensions: number of labels x number of labels.
        testEncodings = [] #This is an array of the encodings of the testSet
        results = [] #This array stores the predictions in encoding form.
        for i in testSet:
            testEncodings.append(decrypt.encode(decrypt, testEncodings[i], attributes))
        results = self.mlp.predict(testEncodings)
        
        #Now, we start a counter. We use the counter as the index of both testSetEncodings and results.
        #Then, we go through and find the index of the 1 in each of them. Use those as the 
        #coordinates in confusionMatrix.
        counter = 0
        while counter < len(testEncodings):
            x = numpy.argmax(testEncodings[counter])
            y = numpy.argmax(results[counter])
            confusionMatrix[x][y] = confusionMatrix[x][y] + 1
            counter = counter+1
        file = open(filename,'w')
        for y in confusionMatrix:
            for x in confusionMatrix[y]:
                file.write(confusionMatrix[x][y])
            file.write('\n')
        return self.mlp.score(testEncodings, labels, sample_weight=None), confusionMatrix
        
    #What do we need to return? How do we process it? How do we send the instances into this, for both training and predicting?