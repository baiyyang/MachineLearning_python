#!/usr/bin python
# -*- coding:utf-8 -*-

'''
线性回归
'''

from numpy import *

def loadDataSet(fileName):
	numFeat = len(open(fileName).readLine().split('\t')) - 1
	dataMat = [];
	labelMat = [];
	fr = open(fileName)
	for line in fr:
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat , labelMat

#计算标准的w，w=(X.T*X).I*X.T*Y
def standRegres(xArr , yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	#计算矩阵的行列式的值
	if linalg.det(xTx) == 0.0:
		print 'This matrix is singular , cannot do inverse'
		return
	ws = xTx.I * (xMat.T * yMat)
	return ws


#局部加权线性回归
def lwlr(testPoint , xArr , yArr , k=1.0):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	m = shape(xMat)[0]
	weights = mat(eye(m))
	for j in range(m):
		diffMat = testPoint - xMat(j , :)
		weights[j , j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0.0:
		print 'This matrix is singular , cannot do inverse'
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

def lwlrTest(testArr , xArr , yArr , k=1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i] , xArr , yArr , k)
	return yHat


#岭回归
def ridgeRegres(xMat , yMat , lam = 2.0):
	xTx = xMat.T * xMat
	denom = xTx + eye(shape(xMat)[1]) * lam
	if linalg.det(denom) == 0.0:
		print 'This matrix is singular , cannot do inverse'
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(xArr , yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	yMean = mean(yMat , 0)
	yMat = yMat - yMean
	xMeans = mean(xMat , 0)
	xVar = var(xMat , 0)
	xMat = (xMat - xMeans)/xVar
	numTestPts = 30
	wMat = zeros((numTestPts , shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat , yMat , exp(i-10))
		wMat[i , :] = ws.T
	return wMat