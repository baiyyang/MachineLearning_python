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

def binSplitDataSet(dataSet , feature , vlaue):
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