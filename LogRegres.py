#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr:
		lineArr = line.strip().split()
		# dataMat 记录的是每个点和他们的初始的权重
		dataMat.append([1.0 , float(lineArr[0]) , float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat , labelMat

def sigmoid(inX):
	return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn , classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	# m,n 是 100 3
	m , n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n,1))
	for k in range(maxCycles):
		# h 和 error 都是向量
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

#随机梯度上升
def stocGradAscent0(dataMatrix , classLabels):
	m , n =shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		# h和error都是数值
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

#改进的随机梯度上升
def stocGradAscent1(dataMatrix , classLabels , numIter = 150):
	m , n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0 + i + j) + 0.01
			randIndex = int(random.uniform(0 , len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights;

def classifyVector(inX , weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5:
		return 1
	else:
		return 0