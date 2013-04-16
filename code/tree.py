import numpy as np

class Node:
    def __init__(self, question, left, right, left_points, right_points):
        self.question = question
        self.left = left
        self.right = right
	self.left_points = left_points
	self.right_points = right_points

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

class Question:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value

    def __repr__(self):
	return 'Question: [%d] <= %f' % (self.feature, self.value)
