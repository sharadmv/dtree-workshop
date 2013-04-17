from train import *
from numpy import *
from scipy.io import loadmat

params = {
    '7.1' :  {
        'trees' : 20,
        'samples' : 350,
        'depth' : 4,
    },
    '6.9': {
        'trees' : 100,
        'samples' : 300,
        'depth' : 5,
        'features' : 20 
    },
    '5.8' : {
        'trees' : 100,
        'samples' : 400,
        'depth' : 30,
        'features' : 6 
    },
    '5.9' : {
        'trees' : 200,
        'samples' : 400,
        'depth' : 30,
        'features' : 5 
    }
}
d = DecisionTree({
    'depth' : 6,
})

r = RandomForest(parameters={
    'trees' : 100,
    'samples' : 1000,
    'depth' : 10,
    'features' : 20 
})

a = AdaBoost(parameters={
    'iterations' : 20,
    'samples' : 1000,
    'depth' : 0,
    'features' : 1 
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
trerr = []
teerr = []
for i in range(1, 201):
    d = RandomForest({
        'trees' : i,
        'samples' : 400,
        'depth' : 30,
        'features' : 5 
    })
    d.train(c)
    training_error = d.test(c)
    test_error = d.test(t)
    trerr.append(training_error)
    teerr.append(test_error)
    print("%f, %f" % (training_error, test_error))

print("Error rate: %f" % d.test(t))

print("Training random forest")
r.train(c)
print("Testing random forest")
print("Error rate: %f" % r.test(t))
"""

print("Training AdaBoost")
a.train(c)
print("Testing AdaBoost")
print("Error rate: %f" % a.test(t))
