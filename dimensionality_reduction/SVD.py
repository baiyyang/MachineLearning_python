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