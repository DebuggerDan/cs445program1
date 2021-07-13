# CS 445, Summer 2021 - Programming Assignment 1 - Dan Jang
# Graph Implementation


import numpy as num
import matplotlib.pyplot as mplot

def graph(n, momentum):
    x, y = num.loadtxt("trainingaccuracy.csv", delimiter = ',', unpack=True)
    x2, y2 = num.loadtxt("testingaccuracy.csv", delimiter = ',', unpack=True)

    mplot.plot(x, y, label = "Accuracy of the Training Set")
    mplot.plot(x2, y2, label = "Accuracy of the Testing Set")

    mplot.title('Hidden Unit(s): ', n, 'Momentum: ', momentum)
    mplot.xlabel('# [Iteration(s)] of Epoch(s)')
    mplot.ylabel('Accuracy (%)')
   
    mplot.legend()
    mplot.show()

    mplotsavefig('plotofaccuracyandepoch.png')