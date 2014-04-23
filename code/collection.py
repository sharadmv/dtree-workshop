import numpy as np
from math import *
from tree import *

class Collection:
    def __init__(self, x, y, distribution=None):
        assert x.shape[0] == y.shape[0], "Mismatching x/y dimensions"
        self.x = x
        self.y = y
        if distribution == None:
            distribution = [] if self.points == 0 else [1.0/self.points]*self.points
        self.distribution = np.array(distribution)

    @property
    def points(self):
        return self.x.shape[0];

    @property
    def features(self):
        return self.x.shape[1];

    def sort(self, feature):
        indices = self.x[:, feature].argsort();
        return Collection(self.x[indices], self.y[indices], self.distribution[indices])

    def entropy(self):
        num_points = self.points
        p = np.average(self.y.transpose()[0], weights=self.distribution)
        q = 1 - p
        if p == 0:
            p = 0.0001
        if q == 0:
            q = 0.0001
        return -p*log(p, 2)-q*log(q, 2)

    def gini(self):
        num_points = self.points
        p = np.average(self.y, weights=self.distribution)
        return 1-p*p

    def partition(self, index):
        dl = self.distribution[:index]
        dl = dl / np.sum(dl)
        dr = self.distribution[index:]
        dr = dr / np.sum(dr)
        return (Collection(self.x[:index], self.y[:index], dl), Collection(self.x[index:], self.y[index:], dr))

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
        return self.entropy()

    def vote(self):
        p = (np.sum(self.y)+0.0)/self.points
        return round(p), p
