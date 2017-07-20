#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
卷积神经网络
'''
from numpy import *

class ConvLayer(object):
	"""docstring for ConvLayer"""
	#参数分别为输入的矩阵的长宽高，卷积核的长宽高，输入矩阵周围的0，步长和学习率
	def __init__(self, input_width , input_height , channel_number , filter_width , fliter_height , 
				 filter_num , zero_padding , stride , activator , learning_rate):		
		self.input_width = input_width
		self.input_height = input_height
		self.channel_number = channel_number
		self.filter_width = filter_width
		self.fliter_height = fliter_height
		self.filter_num = filter_num
		self.zero_padding = zero_padding
		self.stride = stride
		self.output_width = ConvLayer.calculate_output_size(self.input_width , 
			filter_width , zero_padding , stride)
		self.output_height = ConvLayer.calculate_output_size(self.input_height , 
			fliter_height , zero_padding , stride)
		self.output_array = zeros((self.filter_num , self.output_height , self.output_width))
		self.filters = []
		for i in range(filter_num):
			self.filters.append(Filter(filter_width , fliter_height , self.channel_number))
		self.activator = activator
		self.learning_rate = learning_rate

	#确定卷积层输出的大小
	@staticmethod
	def calculate_output_size(input_size , filter_size , zero_padding , stride):
		return (input_size - filter_size + 2 * zero_padding) / stride + 1

	#将误差项传递至上一层
	def bp_sensitivity_map(self , sensitivity_array , activator):
		#处理卷积步长，对原始的sensitivity map进行扩展
		expanded_array = self.expand_sensitivity_map(sensitivity_array)
		#虽然原始输入的zero padding单元也会得到残差
		#但这个残差不需要往上传递，不需要计算
		expanded_width = expanded_array.shape[2]
		zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
		padded_array = padding(expanded_array , zp)
		#初始化delta_array，保存上一层的sensitivity_map
		self.delta_array = self.create_delta_array()
		#对于多个卷积层，传递到上一层的相当于所有的feature map之和
		for f in range(self.filter_num):
			filter = self.filters[f]
			# 将filter的权重翻转180
			flipped_weights = array(map(lambda i : rot90(i , 2) , filter.get_weights()))
			#计算一个filter对应的delta_array
			for d in range(delta_array.shape[0]):
				conv(padded_array[f] , flipped_weights[d] , delta_array[d] , 1 , 0)
			self.delta_array += delta_array
		#将计算结果与激活函数的偏导数做element_wise乘法操作
		derivative = array(self.input_array)
		element_wise_op(derivative.array , activator.backward)
		self.delta_array *= derivative.array


	def expand_sensitivity_map(self , sensitivity_array):
		depth = sensitivity_array.shape[0]
		expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
		expanded_height = (self.input_height - self.fliter_height + 2 * self.zero_padding + 1)
		expanded_array = zeros((depth , expanded_height , expanded_width))
		for i in range(self.output_height):
			for j in range(self.output_width):
				i_pos = i * stride
				j_pos = j * stride
				expanded_array[: , i_pos , j_pos] = sensitivity_array[: , i , j]
		return expanded_array

	def create_delta_array(self):
		return zeros((self.channel_number , self.input_height , self.input_width))

	#计算梯度
	def bp_gradient(self , sensitivity_array):
		expanded_array = self.expand_sensitivity_map(sensitivity_array)
		for f in range(self.filter_num):
			filter = self.filters[f]
			for d in range(filter.weights.shape[0]):
				conv(self.padded_input_array[d] , expanded_array[f] , filter.weights_grad[d] , 1 , 0)
			#计算偏置项的梯度
			filter.bias_grad = expanded_array[f].sum()

	def update(self):
		for filter in self.filters:
			filter.update(self.learning_rate)


#保存了卷积层的参数和梯度，实现了梯度下降算法
#权重随机初始化一个很小的值，偏置初始化为0
class Filter(object):
	"""docstring for Filter"""
	def __init__(self, witdth , height , depth):
		self.weights = random.uniform(-1e-4 , 1e-4 , (depth , height , witdth))
		self.bias = 0
		self.weights_grad = zeros(self.weights.shape)
		self.bias_grad = 0
		
	def __repr__():
		return 'filter weights : %s\tbias : %s' % (repr(self.weights) , repr(self.bias))

	def get_weights(self):
		return self.weights

	def get_bias(self):
		return self.bias

	def update(self , learning_rate):
		self.weights -= learning_rate * self.weights_grad
		self.bias -= learning_rate * self.bias_grad

#实现激活函数
class ReluActivator(object):
	def forward(self , weighted_input):
		return max(0 , weighted_input)

	def backward(self , output):
		return 1 if output > 0 else 0


#卷积层前向计算
def forward(self , input_array):
	self.input_array = input_array
	self.padded_input_array = padding(input_array , self.zero_padding)
	for f in range(self.filter_num):
		filter = self.filters[f]
		conv(self.padded_input_array , filter.get_weights , \
			self.output_array[f] , self.stride , filter.get_bias())
		element_wise_op(self.output_array , self.activator.forward)

#实现对numpy数据进行element wise操作
def element_wise_op(array , op):
	for i in nditer(array , op_flags = ['readwrite']):
		i[...] = op(i)

#计算卷积，自动适配2D和3D的情况
def conv(input_array , kernel_array , output_array , stride , bias):
	#ndim为数组的秩
	channel_number = input_array.ndim
	output_width = output_array.shape[1]
	output_height = output_array.shape[0]
	kernel_width = kernel_array.shape[-1]
	kernel_height = kernel_array.shape[-2]
	for i in range(output_height):
		for j in range(output_width):
			output_array[i][j] = get_patch(input_array , i , j , kernel_width , \
				kernel_height , stride) * kernel_array.sum() + bias


#为数组增加zero padding
def padding(input_array , zp):
	if zp == 0:
		return input_array
	else:
		if input_array.ndim == 3:
			input_width = input_array.shape[2]
			input_height = input_array.shape[1]
			input_depth = input_array.shape[0]
			padded_array = zeros((input_depth , input_height + 2*zp , input_width + 2*zp))
			padded_array[: , zp : zp + input_height , zp : zp + input_width] = input_array
			return padded_array
		elif input_array.ndim == 2:
			input_width = input_array.shape[1]
			input_height = input_array.shape[0]
			padded_array = zeros((input_height + 2*zp , input_width + 2*zp))
			padded_array[zp : zp + input_height , zp : zp + input_width] = input_array
			return padded_array


#max pool层
class MaxPoolingLayer(object):
	"""docstring for MaxPoolingLayer"""
	def __init__(self, input_width , input_height , channel_number , filter_width , 
					fliter_height , stride):
		self.input_width = input_width
		self.input_height = input_height
		self.channel_number = channel_number
		self.filter_width = filter_width
		self.fliter_height = fliter_height
		self.stride = stride
		self.output_width = (input_width - filter_width) / stride + 1
		self.output_height = (input_height - fliter_height) / stride + 1
		self.output_array = zeros((channel_number , output_height , output_width))

	def forward(self , input_array):
		for d in range(self.channel_number):
			for i in range(self.output_height):
				for j in range(self.output_width):
					self.output_array[d,i,j] = (get_patch(input_array[d] , i ,j , \
						self.filter_width , self.height , self.stride).max())

	def backward(self , input_array , sensitivity_array):
		self.delta_array = zeros(input_array.shape)
		for d in range(self.channel_number):
			for i in range(self.output_height):
				for j in range(self.output_width):
					patch_array = get_patch(input_array[d] , i , j , \
						self.filter_width , self.fliter_height , self.stride)
					k , l = get_max_index(patch_array)
					self.delta_array[d , i * stride + k , j * stride + l] = sensitivity_array[d , i , j]
		