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
	# 防止出现概率为0的情况
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainsDocs):
		if trainCategory[i] == 1:
			p1Num += trainmatrix[i]
			p1Denom += sum(trainmatrix[i])
		else:
			p0Num += trainmatrix[i]
			p0Denom += sum(trainmatrix[i])
	p1Vec = log(p1Num/p1Denom)
	p0Vec = log(p0Num/p0Denom)
	return p0Vec , p1Vec , pAbusive

#朴素的贝叶斯分类函数
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts , listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList , postinDoc))
	p0V , p1V , pAb = trainNB0(array(trainMat) , array(listClasses))
	testEntry = ['love' , 'my' , 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList , testEntry))
	print testEntry , 'classified as: ' , classifyNB(thisDoc , p0V , p1V , pAb)