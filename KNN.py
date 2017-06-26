#! /usr/bin python
# -*- coding: UTF-8 -*-
'''
K-Nearset-Neighbor
'''

from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1] , [1.0,1.0] , [0,0] , [0,0.1]])
	labels = ['A' , 'A' , 'B' , 'B']
	return group , labels

def classify0(inX , dataSet , labels , k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX , (dataSetSize , 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel , 0) + 1
	sortedClassCount = sorted(classCount.iteritems() , key = operator.itemgetter(1) , reverse = True)
	return sortedClassCount[0][0]

def normMat(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals , (m,1))
	normDataSet = normDataSet / tile(ranges , (m,1))
	return normDataSet , ranges , minVals

def datingClasstest():
	hoRatio = 0.10
	datingDataMat , datingLabels = file2matrix('datingTestSet.txt')
	normmat , ranges , minvals = normMat(datingDataMat)
	m = normMat.shape[0]
	numsTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numsTestVecs):
		classifierResult = classify0(normMat[i,:] , normMat[numsTestVecs:m , :] , \
			datingLabels[numsTestVecs:m] , 3)
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print 'the total error rate is : %f' % (errorCount/float(numsTestVecs))