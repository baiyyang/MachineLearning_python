#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet(fileName , delim='\t'):
	fr = open(fileName)
	StringArr = [line.strip().split(delim) for line in fr.readLines()]
	datArr = [map(float , line) for line in StringArr]
	return mat(datArr)

#topNfeat是返回原始数据的前N个特征
def pca(dataMat , topNfeat=9999999):
	#去除掉平均值
	meanVals = mean(dataMat , axis = 0)
	meanRemoved = dataMat - meanVals
	#计算协方差矩阵,rowvar=0代表每一行是一个样本
	covMat = cov(meanRemoved , rowvar = 0)
	#eigVals存放特征值(行向量), eigVects存放特征向量,每一列代表一个特征向量
	eigVals , eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat + 1) : -1]
	redEigVects = eigVects[: , eigValInd]
	#低维空间中的数据和重构数据
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat , reconMat