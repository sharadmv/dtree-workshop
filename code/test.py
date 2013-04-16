from train import *
from numpy import *
from scipy.io import loadmat

d = DecisionTree()
r = RandomForest(parameters={
    'trees' : 100,
    'samples' : 100,
    'depth' : 6
})
print("Loading data")
data = loadmat('spamData.mat')
print("Preprocessing")
x = data['Xtrain']
y = data['ytrain']

c = Collection(x, y)

test = data['Xtest']
check = data['ytest']

t = Collection(test, check)

"""
print("Training decision tree")
d.train(c)

print("Error rate: %f" % d.test(t))
"""

print("Training random forest")
r.train(c)
print("Error rate: %f" % r.test(t))
