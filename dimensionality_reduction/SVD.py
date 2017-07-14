#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

def loadExData():
	return [[1,1,1,0,0],
			[2,2,2,0,0],
			[1,1,1,0,0],
			[5,5,5,0,0],
			[1,1,0,2,2],
			[0,0,0,3,3],
			[0,0,0,1,1]]

def svd(data):
	u , sigma , vt = linalg.svd(data)
	return u , sigma , vt

#计算两个向量之间的欧式距离
def euclidSim(inA , inB):
	return 1.0 / (1.0 + linalg.norm(inA - inB))

#计算两个向量之间的皮尔逊相关系数
#向量均为列向量
def pearsSim(inA , inB):
	if len(inA) < 3:
		return 1.0
	return 0.5 + 0.5 * corrcoef(inA , inB , rowvar = 0)[0][1]

#计算两个向量之间的余弦距离
def cosSim(inA , inB):
	num = float(inA.T * inB)
	denom = linalg.norm(inA) * linalg.norm(inB)
	return 0.5 + 0.5 * (num / denom)

#基于物品相似度的推荐引擎
#计算用户对物品的估分评值
#输入分别是数据矩阵，用户编号，相似度计算方法，物品编号
def standEst(dataMat , user , simMeans , item):
	n = shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):
		userRating = dataMat[user , j]
		if userRating == 0:
			continue
		#寻找两个物品都评级的物品
		overLap = nonzero(logical_and(dataMat[: , item].A >0 , dataMat[: , j].A > 0))[0]
		if len(overLap) == 0:
			similarity = 0
		else:
			similarity = simMeans(dataMat[overLap , item] , dataMat[overLap , j])
		print 'the %d and %d similarity is: %f' % (item , j , similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal

def recommend(dataMat , user , N = 3 , simMeans = cosSim , estMethod = standEst):
	#寻找未评级的物品列表
	unratedItems = nonzero(dataMat[user , :].A == 0)[1]
	if len(unratedItems) == 0:
		return 'You rated everything'
	itemScores = []
	for item in unratedItems:
		estimatedScore = estMethod(dataMat , user , simMeans , item)
		itemScores.append((item , estimatedScore))
	return sorted(itemScores , key=lambda jj : jj[1] , reverse=True)[:N]

#基于SVD的评分估计
def svdEst(dataMat , user , simMeans , item):
	n = shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	U , Sigma , VT = linalg.svd(dataMat)
	Sig4 = mat(eye(4) * Sigma[:4])
	#构建转换后的矩阵
	xformedItems = dataMat.T * U[: , :4] * Sig4.I
	for j in range(n):
		userRating = dataMat[user , j]
		if userRating == 0 or j == item:
			continue
		similarity = simMeans(xformedItems[item , :].T , xformedItems[j , :].T)
		print 'the %d and %d similarity is: %f' % (item , j , similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal