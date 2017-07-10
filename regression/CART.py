#!/usr/bin/python
# -*-coding:utf-8 -*-

from numpy import *

class TreeNode(object):
	"""docstring for TreeNode"""
	def __init__(self, feat , val , right , left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float , curLine)
		dataMat.append(fltLine)
	return dataMat

#feature为特征，即列信息
def binSplitDataSet(dataSet , feature , value):
	mat0 = dataSet[nonzero(dataSet[: , feature] > value)[0] , :][0]
	mat1 = dataSet[nonzero(dataSet[: , feature] <= value)[0] , :][0]
	return mat0 , mat1

def createTree(dataSet , leafType = regLeaf , errType = regErr , ops=(1,4)):
	feat , val = chooseBestSplit(dataSet , leafType , errType , ops)
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet , rSet = binSplitDataSet(dataSet , feat , val)
	retTree['left'] = createTree(lSet , leafType , errType , ops)
	retTree['right'] = createTree(rSet , leafType , errType , ops)
	return retTree

#回归树的切分函数
def regLeaf(dataSet):
	return mean(dataSet[: , -1])

#var函数计算均方差，乘以样本总数
def regErr(dataSet):
	return var(dataSet[: , -1]) * shape(dataSet)[0]

#ops分别记录误差的阈值和分类的最小的误差值
def chooseBestSplit(dataSet , leafType=regLeaf , errType=regErr , ops=(1,4)):
	tolS = ops[0]
	tolN = ops[1]
	#如果所有值相等则退出
	if len(set(dataSet[: , -1].T.tolist()[0])) == 1:
		return None , leafType(dataSet)
	m , n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf
	bestIndex = 0
	bestValue = 0
	for featIndex in range(n-1):
		for splitVal in set(dataSet[: , featIndex]):
			mat0 , mat1 = binSplitDataSet(dataSet , featIndex , splitVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	#如果误差减小值小于阈值，退出
	if (S - bestS) < tolS:
		return None , leafType(dataSet)
	mat0 , mat1 = binSplitDataSet(dataSet , bestIndex , bestValue)
	#切分的侯的数据集过小则退出
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None , leafType(dataSet)
	return bestIndex , bestValue

#回归树后剪枝
def isTree(obj):
	return (type(obj).__name__=='dict')

def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

def prune(tree , testData):
	if shape(testData)[0] == 0:
		return getMean(tree)
	if isTree(tree['right']) or isTree(tree['left']):
		lSet , rSet = binSplitDataSet(testData , tree['spInd'] , tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'] , lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'] , rSet)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet , rSet = binSplitDataSet(testData , tree['spInd'] , tree['spVal'])
		errorNoMerge = sum(power(lSet[: , -1] - tree['left'] , 2)) + sum(power(rSet[: , -1] - tree['right'] , 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(power(testData[: , -1] - treeMean , 2))
		if errorMerge < errorNoMerge:
			print 'merging'
			return treeMean
		else:
			return tree
	else:
		return tree 