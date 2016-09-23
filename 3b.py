#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

####BELOW ARE AUXILIARY FUCNTIONS ############
def getNumFeatures(examples):
    return len(examples[0])

def normDistance(a, b):
    #a and b are two points of dimention numFeatures
    mysum = 0.0
    for f,v in b.items():
        diff = a.get(f,0) - v
        mysum += (diff*diff)
    return float(mysum)
def getRandomCentroids(numFeatures, K):
    ## iniatialize centroids randomly
    #return a list (length K), each is a vector(represented, with dimention=numFeatures
    centroids = []
    #random.seed(42)
    for i in range(0,K):
        mu = {}
        for item in range(0,numFeatures):
            mu[item] = random.randint(1,100)   #random interger from 1 to 100
        centroids.append(mu)
    return centroids
def shouldStop(oldCentroids, centroids, iterations, maxIters):
    if iterations > maxIters:
        return True
    return oldCentroids == centroids

def getLabels(examples, centroids):
    #return an assignment zi for eacht data point (examples[i])
    # assignments[i] = j
    assignments = [0]*len(examples)
   
    for data in examples:
        #assignments[examples.index(data)] = 5
        assignments[examples.index(data)] = argmin(data, centroids)
    return assignments

#return the arg min(k = 1, 2, ..., K) ||phi(data)-mu_k || ^ 2
def argmin(data, centroids):
    tmpDist = [0]*len(centroids)
    for  k in range(0, len(centroids)):
        tmpDist[k] = normDistance(data, centroids[k])
    return tmpDist.index(min(tmpDist))

def getCentroids(examples, labels, K):
    centroids = []   #a length-K list of dict (with dimension numFeatures each)
    for k in range(0, K):
        counter = 0
        mysum = {}
        for data in examples:
            mysum[data.get(0)] = 0.0
        average = {}
        for data in examples:
            if labels[examples.index(data)] == k:
                counter+=1
                mysum = summation(mysum, data)
        average = division(mysum, counter)
        centroids.append(average)
    return centroids

def summation(a,b):
    mysum = {}
    for k,v in b.items():
        mysum[k] = a.get(k,0) + v
    return mysum
def division(data,n):
    result = {}
    for k,v in data.items():
        if n != 0:
            result[k] = float(v)/float(n)
        else:
            result[k] = float(v)/random.randint(1,100)
    return result
##################K-MEANS ALGORITHM#######################
def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    random.seed(97)
    ######INITIALIZATION##########################
    numFeatures = getNumFeatures(examples)
    centroids = getRandomCentroids(numFeatures, K)
    print "initial centroids: "
    print centroids
    iterations = 0
    oldCentroids = None
    assignments = []
    print "start updating"
    #######START UPDATING UNTIL CONRVERGENCE######
    while not shouldStop(oldCentroids, centroids, iterations, maxIters):
        oldCentroids = centroids
        iterations += 1
        print "The " + str(iterations) + " iteration"
        assignments = getLabels(examples, centroids)###assignments is a list of length len(examples), each element zi is the assignment of examples[i]
        centroids = getCentroids(examples, assignments, K)  #based on assignments
    print "CONVERGENCE ACHIEVED"
    loss = 0.0
    for i in range(0, len(examples)):
        a = examples[i]  # a data
        b = centroids[assignments[i]] 
        loss += normDistance(a,b)
        #print len(examples)
    loss = float(loss) / float(len(examples))
        #loss += normDistance(examples[i], centroids[examples[i]])
    ###when convergence is achieved, return the final assignments, final centroids and the final loss##########
    return (centroids, assignments, loss)

