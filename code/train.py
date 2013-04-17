from tree import *
from math import *

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
	self.num_learners = 100
	self.weights = []
	self.classifiers = []
	self.learners = [DecisionTree(self.parameters) for _ in range(self.num_learners)]
    
    def train(self, collection):
        points = collection.points
        self.distribution = [1.0 / points] * points
        iterations = self.parameters['iterations']
	for i in range(self.num_learners):
	    print("Training learner: %d" % (i + 1))	
	    k = np.random.choice(range(collection.points), self.parameters['samples'])
	    data = Collection(collection.x[k], collection.y[k])
	    self.learners[i].train(data)    

        for t in range(iterations):
	    print("Iteration %d" % (t+1))
	    best, learner = (1, []), None
	    for l in self.learners:
	        e = self.get_error(l, collection, self.distribution) 
                if e[0] < best[0]:
		    best = e
		    learner = l
            error, correct = best
	    print(len(correct))
	    print("Best error: %f" %  error)
	    a = 1.0/2*np.log((1 - error)/error)
	    print("Error: %f" % (error))
            self.classifiers.append(learner)
	    self.weights.append(a)
	    sum = 0.0
            for i in range(points):
		if correct[i]:
	            self.distribution[i] = self.distribution[i]*exp(-a * t)
		else:
	            self.distribution[i] = self.distribution[i]*exp(a * t)
		sum += self.distribution[i]
            self.distribution = map(lambda x : x / sum, self.distribution)
	      
    def get_error(self, learner, collection, weights):
	sum = 0.0
        correct = []
	for i in range(collection.points):
	    if learner.classify(collection.x[i])[0] == collection.y[i][0]:
		sum += weights[i]
		correct.append(True)
	    else:
		correct.append(False)
	return 1 - sum, correct
    
    def classify(self, sample):
	sum = 0.0
	for learner, weight in zip(self.classifiers, self.weights):
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

class Collection:
    def __init__(self, x, y, available=None):
        assert x.shape[0] == y.shape[0], "Mismatching x/y dimensions"
        self.x = x
        self.y = y

    @property
    def points(self):
        return self.x.shape[0];

    @property
    def features(self):
        return self.x.shape[1];

    def sort(self, feature):
        indices = self.x[:, feature].argsort();
        return Collection(self.x[indices], self.y[indices])

    def entropy(self):
        num_points = self.points
        p = (np.sum(self.y)+0.0)/num_points
        q = 1 - p
        if p == 0:
            p = 0.0001
        if q == 0:
            q = 0.0001
        return -p*log(p, 2)-q*log(q, 2)

    def gini(self):
        num_points = self.points
        p = np.sum(self.y)/(num_points + 0.0)
        return 1-p*p

    def partition(self, index):
        return (Collection(self.x[:index],self.y[0:index]),Collection(self.x[index:],self.y[index:]))

    def empty(self):
        return self.points == 0

    def homogeneous(self):
        s = np.sum(self.y)
        return s == self.points or s == 0

    def split(self,features=None):
        information = self.impurity()
        best = float('-inf')
        split, question = None, None
	if features != None:
	    available = list(range(self.features))
	    np.random.shuffle(available)
	    available = set(available[:features])
        for i in range(self.features):
            sorted_collection = self.sort(i)
	    if features != None and i not in available:
                continue
	    unique = np.unique(sorted_collection.x[:, i])
	    current, counter = 0, 0
            while current < unique.shape[0]:
            	num = unique[current]
                while counter < sorted_collection.points and sorted_collection.x[counter, i] == num:
		    counter+=1
                left, right = sorted_collection.partition(counter)  
                p = (left.points + 0.0)/self.points
                gain = information - p*left.impurity() - (1 - p)*right.impurity()
                if gain >= best:
                    best = gain
                    split = (left, right)
                    question = Question(i, num)
 		current += 1
        return split, question, best

    def impurity(self):
        if self.points == 0:
            return 0
        return self.gini()

    def vote(self):
        p = (np.sum(self.y)+0.0)/self.points
	return round(p), p
