import tensorflow as TensorFlow
import os
import numpy as Numpy


def CreateWeightVariable(shape, dev):
    initial = TensorFlow.random_normal(shape, stddev = dev, dtype=TensorFlow.float32)
    return TensorFlow.Variable(initial)

def LoadWeightVariable(dataDictionary, name):
    return TensorFlow.Variable(dataDictionary[name][0], name="weights")

def CreateBiasVariable(shape):
    initial = TensorFlow.constant(0, shape=shape,dtype=TensorFlow.float32)
    return TensorFlow.Variable(initial)

def LoadBiasVariable(dataDictionary, name):
    return TensorFlow.Variable(dataDictionary[name][1], name="biases")

def Convolution2D(x, W):
    return TensorFlow.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def MaxPool2x2(x):
    return TensorFlow.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def CreateNetwork(inputs, keepProbe, isRetrain):
    if isRetrain:
        dataDictionary = Numpy.load("vgg19.npy", encoding='latin1').item()
        convolution1_1Weights = LoadWeightVariable(dataDictionary, "conv1_1")
        convolution1_1Bias = LoadBiasVariable(dataDictionary, "conv1_1")
        convolution1_2Weights = LoadWeightVariable(dataDictionary, "conv1_2")
        convolution1_2Bias = LoadBiasVariable(dataDictionary, "conv1_2")

        convolution2_1Weights = LoadWeightVariable(dataDictionary, "conv2_1")
        convolution2_1Bias = LoadBiasVariable(dataDictionary, "conv2_1")
        convolution2_2Weights = LoadWeightVariable(dataDictionary, "conv2_2")
        convolution2_2Bias = LoadBiasVariable(dataDictionary, "conv2_2")

        convolution3_1Weights = LoadWeightVariable(dataDictionary, "conv3_1")
        convolution3_1Bias = LoadBiasVariable(dataDictionary, "conv3_1")
        convolution3_2Weights = LoadWeightVariable(dataDictionary, "conv3_2")
        convolution3_2Bias = LoadBiasVariable(dataDictionary, "conv3_2")
        convolution3_3Weights = LoadWeightVariable(dataDictionary, "conv3_3")
        convolution3_3Bias = LoadBiasVariable(dataDictionary, "conv3_3")
        convolution3_4Weights = LoadWeightVariable(dataDictionary, "conv3_4")
        convolution3_4Bias = LoadBiasVariable(dataDictionary, "conv3_4")

        convolution4_1Weights = LoadWeightVariable(dataDictionary, "conv4_1")
        convolution4_1Bias = LoadBiasVariable(dataDictionary, "conv4_1")
        convolution4_2Weights = LoadWeightVariable(dataDictionary, "conv4_2")
        convolution4_2Bias = LoadBiasVariable(dataDictionary, "conv4_2")
        convolution4_3Weights = LoadWeightVariable(dataDictionary, "conv4_3")
        convolution4_3Bias = LoadBiasVariable(dataDictionary, "conv4_3")
        convolution4_4Weights = LoadWeightVariable(dataDictionary, "conv4_4")
        convolution4_4Bias = LoadBiasVariable(dataDictionary, "conv4_4")

        convolution5_1Weights = LoadWeightVariable(dataDictionary, "conv5_1")
        convolution5_1Bias = LoadBiasVariable(dataDictionary, "conv5_1")
        convolution5_2Weights = LoadWeightVariable(dataDictionary, "conv5_2")
        convolution5_2Bias = LoadBiasVariable(dataDictionary, "conv5_2")
        convolution5_3Weights = LoadWeightVariable(dataDictionary, "conv5_3")
        convolution5_3Bias = LoadBiasVariable(dataDictionary, "conv5_3")
        convolution5_4Weights = LoadWeightVariable(dataDictionary, "conv5_4")
        convolution5_4Bias = LoadBiasVariable(dataDictionary, "conv5_4")

        fullyConnected6Weights = CreateWeightVariable([2048, 4096], 0.01)
        fullyConnected6Bias = CreateBiasVariable([4096])

        fullyConnected7Weights = LoadWeightVariable(dataDictionary, "fc7")
        fullyConnected7Bias = LoadBiasVariable(dataDictionary, "fc7")

        fullyConnected8Weights = CreateWeightVariable([4096, 10], 0.01)
        fullyConnected8Bias = CreateBiasVariable([10])
    else:
        convolution1_1Weights = CreateWeightVariable([3, 3, 3, 64], 0.03)
        convolution1_1Bias = CreateBiasVariable([64])
        convolution1_2Weights = CreateWeightVariable([3, 3, 64, 64], 0.03)
        convolution1_2Bias = CreateBiasVariable([64])

        convolution2_1Weights = CreateWeightVariable([3, 3, 64, 128], 0.03)
        convolution2_1Bias = CreateBiasVariable([128])
        convolution2_2Weights = CreateWeightVariable([3, 3, 128, 128], 0.03)
        convolution2_2Bias = CreateBiasVariable([128])

        convolution3_1Weights = CreateWeightVariable([3, 3, 128, 256], 0.03)
        convolution3_1Bias = CreateBiasVariable([256])
        convolution3_2Weights = CreateWeightVariable([3, 3, 256, 256], 0.03)
        convolution3_2Bias = CreateBiasVariable([256])
        convolution3_3Weights = CreateWeightVariable([3, 3, 256, 256], 0.03)
        convolution3_3Bias = CreateBiasVariable([256])
        convolution3_4Weights = CreateWeightVariable([3, 3, 256, 256], 0.03)
        convolution3_4Bias = CreateBiasVariable([256])

        convolution4_1Weights = CreateWeightVariable([3, 3, 256, 512], 0.03)
        convolution4_1Bias = CreateBiasVariable([512])
        convolution4_2Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution4_2Bias = CreateBiasVariable([512])
        convolution4_3Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution4_3Bias = CreateBiasVariable([512])
        convolution4_4Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution4_4Bias = CreateBiasVariable([512])

        convolution5_1Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution5_1Bias = CreateBiasVariable([512])
        convolution5_2Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution5_2Bias = CreateBiasVariable([512])
        convolution5_3Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution5_3Bias = CreateBiasVariable([512])
        convolution5_4Weights = CreateWeightVariable([3, 3, 512, 512], 0.03)
        convolution5_4Bias = CreateBiasVariable([512])

        fullyConnected6Weights = CreateWeightVariable([2048, 4096], 0.01)
        fullyConnected6Bias = CreateBiasVariable([4096])

        fullyConnected7Weights = CreateWeightVariable([4096, 4096], 0.01)
        fullyConnected7Bias = CreateBiasVariable([4096])

        fullyConnected8Weights = CreateWeightVariable([4096, 10], 0.01)
        fullyConnected8Bias = CreateBiasVariable([10])

    convolution1_1 = TensorFlow.nn.relu(Convolution2D(inputs, convolution1_1Weights) + convolution1_1Bias)
    convolution1_2 = TensorFlow.nn.relu(Convolution2D(convolution1_1, convolution1_2Weights) + convolution1_2Bias)
    pool1 = MaxPool2x2(convolution1_2)

    convolution2_1 = TensorFlow.nn.relu(Convolution2D(pool1, convolution2_1Weights) + convolution2_1Bias)
    convolution2_2 = TensorFlow.nn.relu(Convolution2D(convolution2_1, convolution2_2Weights) + convolution2_2Bias)
    pool2 = MaxPool2x2(convolution2_2)

    convolution3_1 = TensorFlow.nn.relu(Convolution2D(pool2, convolution3_1Weights) + convolution3_1Bias)
    convolution3_2 = TensorFlow.nn.relu(Convolution2D(convolution3_1, convolution3_2Weights) + convolution3_2Bias)
    convolution3_3 = TensorFlow.nn.relu(Convolution2D(convolution3_2, convolution3_3Weights) + convolution3_3Bias)
    convolution3_4 = TensorFlow.nn.relu(Convolution2D(convolution3_3, convolution3_4Weights) + convolution3_4Bias)
    pool3 = MaxPool2x2(convolution3_4)

    convolution4_1 = TensorFlow.nn.relu(Convolution2D(pool3, convolution4_1Weights) + convolution4_1Bias)
    convolution4_2 = TensorFlow.nn.relu(Convolution2D(convolution4_1, convolution4_2Weights) + convolution4_2Bias)
    convolution4_3 = TensorFlow.nn.relu(Convolution2D(convolution4_2, convolution4_3Weights) + convolution4_3Bias)
    convolution4_4 = TensorFlow.nn.relu(Convolution2D(convolution4_3, convolution4_4Weights) + convolution4_4Bias)
    pool4 = MaxPool2x2(convolution4_4)

    convolution5_1 = TensorFlow.nn.relu(Convolution2D(pool4, convolution5_1Weights) + convolution5_1Bias)
    convolution5_2 = TensorFlow.nn.relu(Convolution2D(convolution5_1, convolution5_2Weights) + convolution5_2Bias)
    convolution5_3 = TensorFlow.nn.relu(Convolution2D(convolution5_2, convolution5_3Weights) + convolution5_3Bias)
    convolution5_4 = TensorFlow.nn.relu(Convolution2D(convolution5_3, convolution5_4Weights) + convolution5_4Bias)
    reshape = TensorFlow.reshape(convolution5_4, [-1, 2*2*512])

    fullyConnected6 = TensorFlow.nn.dropout(TensorFlow.nn.relu(TensorFlow.matmul(reshape, fullyConnected6Weights) + fullyConnected6Bias), keepProbe)
    fullyConnected7 = TensorFlow.nn.dropout(TensorFlow.nn.relu(TensorFlow.matmul(fullyConnected6, fullyConnected7Weights) + fullyConnected7Bias), keepProbe)
    fullyConnected8 = TensorFlow.nn.relu(TensorFlow.matmul(fullyConnected7, fullyConnected8Weights) + fullyConnected8Bias)

    return fullyConnected8
