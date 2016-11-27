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
        return 1
    def classify(self, testSetEncodings, attributes, labels):
        #In order to classify, we use predict given the test set.
        #X is a 2D array of the encodings of the testSet, and the list of attributes.
        decrypt = encoder
        results = []
        for i in testSetEncodings:
            results.append(decrypt.encode(decrypt, testSetEncodings[i], attributes))
        self.mlp.predict(results)
        return self.mlp.score(results, labels, sample_weight=None)
        
    #What do we need to return? How do we process it? How do we send the instances into this, for both training and predicting?