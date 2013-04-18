from tree import *
from math import *
from collection import *

import numpy as np

class Classifier:
    def __init__(self, parameters):
        self.parameters = parameters

    def test(self, collection):
	sum = 0.0
	for i in range(collection.points):
	    if self.classify(collection.x[i])[0] == collection.y[i][0]:
		sum += 1
	return (1 - (sum)/collection.points)

class AdaBoost(Classifier):
    def __init__(self, parameters):
        Classifier.__init__(self, parameters)
	self.weights = []
	self.learners = []
    
    def train(self, collection):
        iterations = self.parameters['iterations']
        feature_list = list(range(collection.features))
	for _ in range(iterations):
	    print("Training learner: %d" % (_ + 1))
	    np.random.shuffle(feature_list)
	    learner = WeakDecisionTree({
	        'depth' : self.parameters['depth'],
                'feature_list' : feature_list[:self.parameters['num_features']] 
	    })
            learner.train(collection)
	    self.learners.append(learner)
        points = collection.points
        self.distribution = [1.0 / points] * points

        for t in range(iterations):
	    learner = self.learners[t]
	    print("Iteration %d" % (t+1))
            error, correct = self.get_error(learner, collection)
	    a = 1.0/2*np.log((1 - error)/error)
	    print("Error: %f" % (error))
            self.learners.append(learner)
	    self.weights.append(a)
	    sum = 0.0
            for i in range(points):
		if correct[i]:
	            self.distribution[i] = self.distribution[i]*exp(-a * t)
		else:
	            self.distribution[i] = self.distribution[i]*exp(a * t)
		sum += self.distribution[i]
            self.distribution = map(lambda x : x / sum, self.distribution)
	      
    def get_error(self, learner, collection):
	sum = 0.0
        correct = []
	for i in range(collection.points):
	    if learner.classify(collection.x[i])[0] == collection.y[i][0]:
		sum += collection.distribution[i]
		correct.append(True)
	    else:
		correct.append(False)
	return 1 - sum, correct
    
    def classify(self, sample):
	sum = 0.0
	for learner, weight in zip(self.learners, self.weights):
	    output = learner.classify(sample)[0]
	    if output == 0:
		output = -1
	    sum += weight * output
	val = np.sign(sum)
	return (0, sum) if val == -1 else (1, sum)
        

class RandomForest(Classifier):
    def __init__(self, parameters):
        Classifier.__init__(self, parameters)	
        self.forest = []

    def train(self, collection):
        for _ in range(self.parameters['trees']):
            tree = DecisionTree(self.parameters)
            k = np.random.choice(range(collection.points), self.parameters['samples'])
            data = Collection(collection.x[k], collection.y[k])
	    print("Training tree: %d" % (_ + 1))
            tree.train(data) 
	    self.forest.append(tree)
    
    def classify(self, sample):
	sum = 0.0
	num_trees = self.parameters['trees']
	for tree in self.forest:
	    sum += tree.classify(sample)[1]
	return (round(sum/num_trees), sum/num_trees)
	
class DecisionTree(Classifier):
    def __init__(self, parameters={}):
        self.tree = None
        self.trained = False
	self.parameters = parameters
        if 'features' not in parameters:
	    self.parameters['features'] = None
        if 'distribution' not in parameters:
	    self.parameters['distribution'] = None

    def train(self, collection):
        self.tree = self.generate_tree(collection)
        self.trained = True

    def generate_tree(self, collection, depth=0):
        if collection.points == 0 or collection.homogeneous() or depth > self.parameters['depth']:
            node = Leaf(collection)
            return node
        (left, right), question, gain = collection.split(self.parameters['features'])
        if gain == 0:
            node = Leaf(collection)
            return node
        left_child = self.generate_tree(left, depth + 1)
        right_child = self.generate_tree(right, depth + 1)
        node = Node(question, left_child, right_child, left, right)
	node.gain = gain
        return node

    def decide(self, sample, tree):
        if isinstance(tree, Leaf):
            return tree.collection.vote()
        else:
            question = tree.question
            val = sample[question.feature]
            if val > question.value:
                return self.decide(sample, tree.right)
            else:
                return self.decide(sample, tree.left)

    def classify(self, sample):
        if not self.trained:
            return "Hasn't been trained yet"
        else:
            return self.decide(sample, self.tree)

class WeakDecisionTree(DecisionTree):
    def __init__(self, parameters):
        DecisionTree.__init__(self, parameters)

    def train(self, collection):
	print("Using parameters:", self.parameters['feature_list'])
	data = Collection(collection.x[:,self.parameters['feature_list']], collection.y, collection.distribution)
        self.tree = self.generate_tree(data)
        self.trained = True
