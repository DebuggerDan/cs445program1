# CS 445, Summer 2021 - Programming Assignment 1 - Dan Jang
# Two-Layer Neural Net Perceptron for MNIST Input for 784 Inputs & 10 Outputs, 1 Hidden Layer


import numpy as nump
import csv
from sklearn.metrics import confusion_matrix
import graph
# import tensorflow


# Setup: MNIST Data Input

trainfile = open("mnist_train.csv", "r")
trainset = csv.reader(trainfile)
traindata = nump.array(list(trainset))

testfile = open("mnist_test.csv", "r")
testset = csv.reader(testfile)
testdata = nump.array(list(testset))


# Setup: Variables & Parameters for Experiments #1 and #2

inputs = 784
# The numpber of inputs!
momentum = 0.9
# The momentum!
learnrate = 0.1
# The learning rate!
epochs = 20
# The numpber of epoch(s)! [E.g. 'cycles']
n = 100
# n, hidden units
bi = 1
# The bias!
confusion = nump.zeros((10, 10), dtype = int)
# The confusion matrix constant!


# Setup: Sigmoid Function [Classification: Activation Function]

def activationsigmoid(n):
    return 1 / (1 + nump.exp(-n))

activations = nump.zeros((1, n + 1))
activations[0,0] = 1


# Setup: Weights

nw = -0.05
pw = 0.05

weightinput = nump.random.uniform(nw, pw,(inputs, n))
pweightinput = nump.zeros((inputs, n))

weightoutput = nump.random.uniform(nw, pw, (n + 1, 10))
pweeighoutput = nump.zeros((n + 1, 10))


# Pre-Start: Arrays for runtime data array(s) for both training and test runs (epochs/experiments)

# prep = 255

trainrundata = nump.asfarray(traindata[:, 1:]) / 255
trainlabeldata = nump.asfarray(traindata[:, 1:])

testrundata = nump.asfarray(testdata[:, 1:]) / 255
testlabeldata = nump.asfarray(testdata[:, 1:])


# Pre-Start: Task Settings (By k-th unit)

k = 0.9
other = 0.1

target = nump.arange(10)

traintk = (target == trainlabeldata).astype(nump.float)
traintk[traintk == 0] = other
traintk[traintk == 1] = k

testtk = (target == testlabeldata).astype(nump.float)
testtk[testtk == 0] = other
testtk[testtk == 1] = k


# Core: Two-Layer Perceptron

def twolayerperceptron(epoch, label, dataset, task, config):
    global confusion, inputs, epochs, pweightinput, weightinput, pweightoutput, weightoutput
    # Perceptron Globals

    actuala = []
    # Array for actual value(s) storage
    predicta = []
    # Array for predictied value(s) storage

    for idx in range(datset.shape[0]): # Loop for Training & Testing Datasets
        
        # Pre-Start (Loop): Initializing target values, loading datasets, then reshaping dataset.

        tvalue = label[idx, 0].astype('int')
        actualaappend(tvalue)
        x = dataset[idx]
        x[0] = bi
        x = x.reshape(1, inputs)

        # Start (Loop): Activation of the Output and Hidden Layers

        zhidden = nump.dot(x, weightinput)
        sigmoidhidden = activationsigmoid(zhidden)
        
        activations[0, 1:] = sigmoidhidden

        zoutput = nump.dot(activations, weightoutput)
        sigmoidoutput = activationsigmoid(zoutput)

        # Start (Loop): Finding the maximum argument and then appending it to predict-array

        prediction = nump.argmax(sigmoidoutput)
        predicta.append(prediction)

        # Intensive (Training Perceptrons)
        if config == 1 and epoch > 0: # If we are currently on k+1'th epoch [cycle] and config has been enabled
            outputerror = sigmoidoutput * (1 - sigmoidoutput) * (task[idx] - sigmoidoutput)
            hiddenerror = sigmoidhidden * (1 - sigmoidhidden) * nump.dot(outputerror, weightoutput[1:,:].T)

            deltainput = (learnrate * hiddenerror * x.T) + (momentum * pweightinput)
            pweighinput = deltainput
            weightinput = weightinput + deltainput

            deltaoutput = (learnrate * outputerror * activations.T) + (momentum * pweightoutput)
            pweightoutput = deltaoutput
            weightoutput = weightoutput + deltaoutput

        
    # Start: Accuracy Measurement
    accuracyv = (nump.array(predicta) == nump.array(actuala)).sum() / float(len(actuals))
    acc = accuracyv * 100

    if config == 1 and epoch == (epochs - 1): # If we are currently testing & If testing is complete
        genconfusion = confusion_matrix(actuala, predicta)
        confusion = nump.add(genconfusion, confusion)
        print("Epoch #", epochs, " - Confusion Matrix")
        print(confusion)

    if config == 1 and epoch != (epochs - 1): # If we are currently testing & If testing is not complete
        genconfusion = confusion_matrix(actuala, predicta)
        confusion = nump.add(genconfusion, confusion)

    return acc

def setaccuracy(accvalue, acc, dataset):
    with open(dataset, 'a') as currset:
        appender = csv.writer(currset)
        appender.writerow([accvalue, acc])

# Post-Training/Testing: Print Experimental Values
print('Current # of hidden units: ', n, ', Momentum is at: ', momentum)
print('Training & Test Sets (Respectively): ', trainrundata, testrundata)

for idx2 in range(epochs):
    trainingsetaccuracy = twolayerperceptron(idx2, trainrundata, trainlabeldata, traintk, 0)
    testingsetaccuracy = twolayerperceptron(idx2, testrundata, trainlabeldata, testtk, 1)

graph.graph(n, momentum)