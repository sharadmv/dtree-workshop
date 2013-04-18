from classifier import *
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
    'trees' : 200,
    'samples' : 400,
    'depth' : 30,
    'features' : 5 
})

a = AdaBoost(parameters={
    'iterations' : 20,
    'depth' : 2,
})

print("Loading data")
data = loadmat('spamData.mat')
x = data['Xtrain']
y = data['ytrain']
c = Collection(x, y)

xtest = data['Xtest']
ytest = data['ytest']

t = Collection(xtest, ytest)

trerr = []
teerr = []
for i in [1, 10, 50, 100]:
    print("Training AdaBoost %d" % i)
    a = AdaBoost(parameters={
        'iterations' : i,
        'depth' : 1,
    })
    a.train(c)
    training_error = a.test(c)
    test_error = a.test(t)
    trerr.append(training_error)
    teerr.append(test_error)
    print("%f, %f" % (training_error, test_error))
