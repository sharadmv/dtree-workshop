from train import *
from numpy import *
from scipy.io import loadmat

x = array([[1,2],[2,4],[3,3],[4,5]])
y = array([[0],[0],[1],[1]])
c = Collection(x, y)
d = DecisionTree()
d.train(c)

print("Loading data")
data = loadmat('spamData.mat')
print("Preprocessing")
x = data['Xtrain']
y = data['ytrain']
c = Collection(x, y)
print("Training")
d.train(c)
