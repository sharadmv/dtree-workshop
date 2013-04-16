import numpy as np

class Node:
    def __init__(self, question, left, right):
        self.question = question
        self.left = left
        self.right = right

    def get_question(self):
        return self.question

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

class Leaf:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self):
        return self.collection

    def vote(self):
        p = (np.sum(self.collection.y)+0.0)/self.collection.points
        if p > 0.5:
            return (1, p)
        else:
            return (0, 1- p)

class Question:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value
