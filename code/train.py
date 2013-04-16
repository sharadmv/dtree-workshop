from tree import *
import numpy as np
from math import *

class DecisionTree:
    def __init__(self, parameters={}):
        self.tree = None
        self.trained = False
        self.parameters = {}

    def train(self, collection):
        self.tree = self.generate_tree(collection)
        self.trained = True

    def generate_tree(self, collection, depth=0):
        if depth > 4 or collection.points == 0 or collection.homogeneous():
            node = Leaf(collection)
            return node
        print("Determining split at depth: %d" % depth)
        (left, right), question, gain = collection.split()
        left_child = self.generate_tree(left, depth + 1)
        right_child = self.generate_tree(right, depth + 1)
        node = Node(question, left_child, right_child)
        return node

    def decide(self, sample, tree):
        if isinstance(tree, Leaf):
            return tree.vote()
        else:
            question = tree.question
            val = sample[question.feature]
            if val >= question.value:
                return self.decide(sample, tree.right)
            else:
                return self.decide(sample, tree.left)

    def classify(self, sample):
        if not self.trained:
            return "Hasn't been trained yet"
        else:
            return self.decide(sample, self.tree)

class Collection:
    def __init__(self, x, y):
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
        p = np.sum(self.y)
        return 1-p*p

    def partition(self, index):
        return (Collection(self.x[:index],self.y[0:index]),Collection(self.x[index:],self.y[index:]))

    def empty(self):
        return self.points == 0

    def homogeneous(self):
        s = np.sum(self.y)
        return s == self.points or s == 0

    def split(self):
        information = self.impurity()
        best = float('-inf')
        split, question = None, None
        for i in range(self.features):
            sorted_collection = self.sort(i)
            for j in range(sorted_collection.points+1):
                print("Testing feature %d at %d" % (i, j))
                left, right = sorted_collection.partition(j)
                p = (left.points + 0.0)/self.points
                gain = information - p*left.impurity() - (1 - p)*right.impurity()
                if gain > best:
                    best = gain
                    split = (left, right)
                    question = Question(i, self.x[j, i])
        return split, question, best

    def impurity(self):
        if self.points == 0:
            return 0
        return self.entropy()
