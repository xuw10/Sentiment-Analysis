#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    b = x.split()
    cnt = Counter(b)
    mydict = {}
    for item in list(cnt):
        mydict[item] = cnt[item]
    return mydict
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!

    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # dict
    numIters = 20
    eta = 0.04
    phi = {}    #dict
    for data in trainExamples:
        phi = extractWordFeatures(data[0])
        for item in phi.keys():
            weights[item] = 0
    #############the above 4 lines are initialization of weights
    #print "breakpoint 4"
    for (x,y) in trainExamples:
        phi = featureExtractor(x)
    #print "breakpoint 5"
    for i in range (numIters):
        for (x,y) in trainExamples:
            phi = featureExtractor(x)
            
            margin = dotProduct(weights, phi) * y
            if margin < 1:
                increment(weights, eta*y, phi)
    return weights

############################################################
# Problem 2c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) can be anything (randomize!) with a nonzero score under the given weight vector
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        phi ={}  # phi is also a dict data type
        values = weights.values()
        for item, val in weights.items():
            phi[item] = random.randrange(1,100)
        y = 1
        product = 0
        lval = len(values)
        phis = phi.values()
        for j in range(0,lval):
            product += values[j] * phis[j]
        #define product as the dot product of weight vector and phi: product = phi*weights
        if product >= 0:
            y = 1
        else:
            y = -1
                
        print "product:" + str(product) + " y:" + str(y) 
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 2f: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        newx = x.replace(" ", "")
        b = []
        for i in range (0, len(newx)-n+1):
            tmpstr = ""
            for j in range(0,n):
                tmpstr += newx[i+j]
          
            b.append(tmpstr)
        #now need to count the occurrence times of each item in b
        cnt = Counter(b)
        mydict = {}
        for item in list(cnt):
            mydict[item] = cnt[item]
        return mydict
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return extract

############################################################
# Problem 2h: extra credit features

def extractExtraCreditFeatures(x):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3: k-means
############################################################

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




#def kmeans(examples, K, maxIters):
#    '''
#    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
#    K: number of desired clusters
#    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
#    Return: (length K list of cluster centroids,
#            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
#            final reconstruction loss)
#    '''
#    # BEGIN_YOUR_CODE (around 35 lines of code expected)
#    # END_YOUR_CODE
