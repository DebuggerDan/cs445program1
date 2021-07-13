# CS 445, Summer 2021 - Programming Assignment 1 - Dan Jang
# Two-Layer Neural Net Perceptron for MNIST Input for 784 Inputs & 10 Outputs, 1 Hidden Layer


import numpy as num
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
# import tensorflow


# Setup: MNIST Data Input

trainfile = open("mnist_train.csv", "r")
trainset = csv.reader(trainfile)
traindata = num.array(list(traindata))

testfile = open("mnist_test.csv", "r")
testset = csv.reader(testfile)
testdata = num.array(list(testset))


# Setup: Variables & Parameters for Experiments #1 and #2

inputs = 784
# The number of inputs!
momentum = 0.9
# The momentum!
learnrate = 0.1
# The learning rate!
epochs = 20
# The number of epoch(s)! [E.g. 'cycles']
n = 100
# n, hidden units
bi = 1
# The bias!


# Setup: Sigmoid Function [Classification: Activation Function]

def activationsigmoid(n):
    return 1 / (1 + num.exp(-n))


# Setup: Weights

nw = -0.05
pw = 0.05

weightinput = num.random.uniform(nw, pw,(inputs, n))
pweightinput = num.zeros((inputs, n))

weightoutput = num.random.uniform(nw, pw, (n + 1, 10))
pweeighoutput = num.zeros((n + 1, 10))


# Pre-Start: Arrays for runtime data array(s) for both training and test runs (epochs/experiments)

prep = 255

trainrundata = num.asfarray(traindata[:, 1:]) / prep
trainlabeldata = num.asfarray(traindata[:, 1:])

testrundata = num.asfarray(testdata[:, 1:]) / prep
testlabeldata = num.asfarray(testdata[:, 1:])


# Pre-Start: Task Settings (By k-th unit)

k = 0.9
other = 0.1

target = num.arange(10)

traintk = (target == trainlabeldata).astype(num.float)
traintk[traintk == 0] = other
traintk[traintk == 1] = k

testtk = (target == testlabeldata).astype(num.float)
testtk[testtk == 0] = other
testtk[testtk == 1] = k


# Core: Two-Layer Perceptron

