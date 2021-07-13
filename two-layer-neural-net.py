# CS 445, Summer 2021 - Programming Assignment 1 - Dan Jang
# Two-Layer Neural Net Perceptron for MNIST Input for 784 Inputs & 10 Outputs, 1 Hidden Layer


import numpy
import csv
import graph
import matplotlib
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')
# import tensorflow


# Setup: MNIST Data Input

trainfile = open("mnist_train.csv", "r")
trainset = csv.reader(trainfile)
traindata = numpy.array(list(trainset))

testfile = open("mnist_test.csv", "r")
testset = csv.reader(testfile)
testdata = numpy.array(list(testset))


# Setup: Variables & Parameters for Experiments #1 and #2

inputs = 784
# The number of inputs!
momentum = 0.9
# The momentum!
learnrate = 0.1
# The learning rate!
epochs = 50
# The number of epoch(s)! [E.g. 'cycles']
n = 20
# n, hidden units
bi = 1
# The bias!
confusion = numpy.zeros((10, 10), dtype = int)
# The confusion matrix constant!


# Setup: Sigmoid Function [Classification: Activation Function]

def activationsigmoid(n):
    return 1 / (1 + numpy.exp(-n))

activations = numpy.zeros((1, n + 1))
activations[0,0] = 1


# Setup: Weights

nw = -0.05
pw = 0.05

weightinput = numpy.random.uniform(nw, pw,(inputs, n))
pweightinput = numpy.zeros((inputs, n))

weightoutput = numpy.random.uniform(nw, pw, (n + 1, 10))
pweightoutput = numpy.zeros((n + 1, 10))


# Pre-Start: Arrays for runtime data array(s) for both training and test runs (epochs/experiments)

prep = 255

trainrundata = numpy.asfarray(traindata[:, 1:]) / prep
trainlabeldata = numpy.asfarray(traindata[:, :1])

testrundata = numpy.asfarray(testdata[:, 1:]) / prep
testlabeldata = numpy.asfarray(testdata[:, :1])

# For Experiment 2:
# training_exp2 = numpy.asfarray(traindata[:15000]) # A Quarter of Training Data
# training_exp2pt2 = numpy.asfarray(traindata[:30000]) # Half of Training Data



# Pre-Start: Task Settings (By k-th unit)

# k = 0.9
# other = 0.1

target = numpy.arange(10)

traintk = (target == trainlabeldata).astype(float)

traintk[traintk == 0] = 0.1
traintk[traintk == 1] = 0.9

testtk = (target == testlabeldata).astype(float)

testtk[testtk == 0] = 0.1
testtk[testtk == 1] = 0.9


# Core: Two-Layer Perceptron

def twolayerperceptron(epoch, dataset, label, task, config):
    global confusion, inputs, epochs, pweightinput, weightinput, pweightoutput, weightoutput
    # Perceptron Globals

    actuala = []
    # Array for actual value(s) storage
    predicta = []
    # Array for predictied value(s) storage

    for idx in range(dataset.shape[0]): # Loop for Training & Testing Datasets
        
        # Pre-Start (Loop): Initializing target values, loading datasets, then reshaping dataset.

        tvalue = label[idx, 0].astype('int')
        actuala.append(tvalue)
        x = dataset[idx]
        x[0] = bi
        x = x.reshape(1, inputs)

        # Start (Loop): Activation of the Output and Hidden Layers

        zhidden = numpy.dot(x, weightinput)
        sigmoidhidden = activationsigmoid(zhidden)
        
        activations[0, 1:] = sigmoidhidden

        zoutput = numpy.dot(activations, weightoutput)
        sigmoidoutput = activationsigmoid(zoutput)

        # Start (Loop): Finding the maximum argument and then appending it to predict-array

        prediction = numpy.argmax(sigmoidoutput)
        predicta.append(prediction)

        # Intensive (Training Perceptrons)
        if config == 1 and epoch > 0: # If we are currently on k+1'th epoch [cycle] and config has been enabled
            outputerror = sigmoidoutput * (1 - sigmoidoutput) * (task[idx] - sigmoidoutput)
            hiddenerror = sigmoidhidden * (1 - sigmoidhidden) * numpy.dot(outputerror, weightoutput[1:,:].T)

            deltainput = (learnrate * hiddenerror * x.T) + (momentum * pweightinput)
            pweighinput = deltainput
            weightinput = weightinput + deltainput

            deltaoutput = (learnrate * outputerror * activations.T) + (momentum * pweightoutput)
            pweightoutput = deltaoutput
            weightoutput = weightoutput + deltaoutput

        
    # Start: Accuracy Measurement
    accuracyv = (numpy.array(predicta) == numpy.array(actuala)).sum() / float(len(actuala))
    acc = accuracyv * 100

    if config == 1 and epoch == (epochs - 1): # If we are currently testing & If testing is complete
        genconfusion = confusion_matrix(actuala, predicta)
        confusion = numpy.add(genconfusion, confusion)
        print("Epoch #", epochs, " - Confusion Matrix")
        print(confusion)

    if config == 1 and epoch != (epochs - 1): # If we are currently testing & If testing is not complete
        genconfusion = confusion_matrix(actuala, predicta)
        confusion = numpy.add(genconfusion, confusion)

    return acc

def setaccuracy(accvalue, acc, dataset):
    with open(dataset, 'a') as currset:
        appender = csv.writer(currset)
        appender.writerow([accvalue, acc])

# Post-Training/Testing: Print Experimental Values
print('Current # of hidden units: ', n, ', Momentum is at: ', momentum)
print('Training & Test Sets (Respectively): ')
print(trainrundata)
print(testrundata)

for idx2 in range(epochs):
    trainingsetaccuracy = twolayerperceptron(idx2, trainrundata, trainlabeldata, traintk, 0)
    testingsetaccuracy = twolayerperceptron(idx2, testrundata, trainlabeldata, testtk, 1)
    
    setaccuracy(idx2, trainingsetaccuracy, 'trainingaccuracy.csv')
    setaccuracy(idx2, testingsetaccuracy, 'testingaccuracy.csv')

graph.graph(n, momentum)