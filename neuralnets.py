#Josh Parker and Sam Goree's Neural Nets homework #3 
import sklearn as sk

#So, if i understand correctly, we're going to have a main class, an IO class, and this neural nets class?
#We run the main class, which uses the IO class to take a file path and create an array of instances, and then we send them through the neural nets class stuff,
# and then use the results and the IO class to output a file.

#replicate a main class, use iotools.py to create the instances, then use scikitLearn in neuralNets as the algorithm class.
class neuralnets:
    X = [][]#array of size (n_samples, n_features), which holds the training instances
    #as floating point feature vectors.
    Y = [][]#array of size (n_samples), which holds the class labels for the training instances.

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X, y)                         
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
        beta_1=0.9, beta_2=0.999, early_stopping=False,
        epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
        learning_rate_init=0.001, max_iter=200, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
        warm_start=False)
       
    clf.predict(#stuff?)       



    #What do we need to return? How do we process it? How do we send the instances into this, for both training and predicting?