import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"

        n = len(trainingData)
        trainingLabels = np.array(trainingLabels)
        w = np.ones(n) / n
        for m in range(self.boosting_iterations):
            self.classifiers[m].train(trainingData, trainingLabels, w)
            error = 0.0
            pred = np.array(self.classifiers[m].classify(trainingData))
            error += w[trainingLabels != pred].sum()
            f = error / (1 - error)
            w[trainingLabels == pred] *= f
            w /= w.sum()
            self.alphas[m] = -np.log(f)

        # util.raiseNotDefined()

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        # "*** YOUR CODE HERE ***"

        poll = 0
        for m in range(self.boosting_iterations):
            pred = np.array(self.classifiers[m].classify(data))
            poll = np.add(poll, pred * self.alphas[m])
        poll = np.sign(poll)
        poll[poll == 0] = np.random.choice([-1, 1], np.count_nonzero(poll == 0))
        return list(poll)

        # util.raiseNotDefined()