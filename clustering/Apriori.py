#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
	return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	#frozenset 用户不可改变的集合
	return map(frozenset , C1)

#输入分别为数据集和，候选项集列表和最小支持度
def scanD(D , Ck , minSupport):
	ssCnt = {}
	for can in Ck:
		for tid in D:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0 , key)
		supportData[key] = support
	return retList , supportData

#create Ck , 即生成候选项集
def aprioriGen(Lk , k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1 , lenLk):
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()
			#连接步
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])
	return retList

def apriori(dataSet , minSupport = 0.5):
	C1 = createC1(dataSet)
	D = map(set , dataSet)
	L1 , supportData = scanD(D , C1 , minSupport)
	L = [L1]
	k = 2
	while len(L[k-2]) > 0:
		Ck = aprioriGen(L[k-2] , k)
		Lk , supK = scanD(D , Ck , minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L , supportData

#关联规则生成函数
#输入参数分别为频繁项集列表，支持度数据字典和最小置信度
def generateRules(L , supportData , minConf = 0.7):
	bigRuleList = []
	#集合元素大于1个开始
	for i in range(1 , len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if i > 1:
				rulesFromConseq(freqSet , H1 , supportData , bigRuleList , minConf)
			else:
				calcConf(freqSet , H1 , supportData , bigRuleList , minConf)
	return bigRuleList

#对规则进行评估
#freqSet是频繁k项集，H是包含每个集合的频繁K项集，brl是bigRuleList
def calcConf(freqSet , H , supportData , brl , minConf = 0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet - conseq]
		if conf >= minConf:
			print freqSet - conseq , '-->' , conseq , 'conf:' , conf
			brl.append((freqSet - conseq , conseq , conf))
			prunedH.append(conseq)
	return prunedH

#生成候选规则集合
def rulesFromConseq(freqSet , H , supportData , brl , minConf = 0.7):
	m = len(H[0])
	if len(freqSet) > (m + 1):
		Hmp1 = aprioriGen(H , m + 1)
		Hmp1 = calcConf(freqSet , Hmp1 , supportData , brl , minConf)
		if len(Hmp1) > 1:
			rulesFromConseq(freqSet , Hmp1 , supportData , brl , minConf)