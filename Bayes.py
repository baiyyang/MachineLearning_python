#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
	postingList = [['my' , 'dog' , 'has' , 'flea' , 'problems' , 'help' , 'please'], \
					['maybe' , 'not' , 'take' , 'him' , 'to' , 'dog' , 'park' , 'stupid'], \
					['my' , 'dalmation' , 'is' , 'so' , 'cute' , 'I' , 'love' , 'him'], \
					['stop' , 'posting' , 'stupid' , 'worthless' , 'garbage'], \
					['mr' , 'licks' , 'ate' , 'my' , 'steak' , 'how' , 'to' , 'stop' , 'him'], \
					['quit' , 'buying' , 'worthless' , 'dog' , 'food' , 'stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList , classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWords2Vec(vocabList , inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print 'the word: %s is not in my vocabulary!' % word
	return returnVec

# 训练函数
def trainNB0(trainmatrix , trainCategory):
	numTrainsDocs = len(trainmatrix)
	numWords = len(trainmatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainsDocs)
	p0Num = zeros(numWords)
	p1Num = zeros(numWords)
	p0Denom = 0.0
	p1Denom = 0.0
	for i in range(numTrainsDocs):
		if trainCategory[i] == 1:
			p1Num += trainmatrix[i]
			p1Denom += sum(trainmatrix[i])
		else:
			p0Num += trainmatrix[i]
			p0Denom += sum(trainmatrix[i])
	p1Vec = p1Num / p1Denom
	p0Vec = p0Num / p0Denom
	return p0Vec , p1Vec , pAbusive