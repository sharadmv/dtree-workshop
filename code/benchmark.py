from classifier import *
from scipy.io import loadmat

d = DecisionTree({
    'depth' : 9,
})

r = RandomForest(parameters={
    'trees' : 10,
    'samples' : 800,
    'depth' : 5,
    'features' : 4
})

a = AdaBoost(parameters={
    'iterations' : 20,
    'depth' : 2,
})

print("Loading data")
data = loadmat('spamData.mat')
train, test = loadmat('train.mat'), loadmat('test.mat')
x = np.array(train['Xtrain'][:, 0:100].todense())
y = train['ytrain']
c = Collection(x, y)

xtest = np.array(test['Xtest'][:, 0:100].todense())
ytest = test['ytest']

t = Collection(xtest, ytest)

def benchmark(classifier):
    classifier.train(c)
    print("Training Error rate: %f" % classifier.test(c))
    print("Testing Error rate: %f" % classifier.test(t))

print("Training Decision Tree")
#benchmark(d)
print("Training Random Forest")
benchmark(r)
print("Training AdaBoost")
#benchmark(a)
