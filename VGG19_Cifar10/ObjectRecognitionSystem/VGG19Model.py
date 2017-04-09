import os
import tensorflow as TensorFlow

import numpy as Numpy
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG19Model:
    def __init__(self, vgg19NumpyPath=None):
        if vgg19NumpyPath is None:
            path = inspect.getfile(VGG19Model)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19NumpyPath = path
            print(vgg19NumpyPath)

        self.dataDictionary = Numpy.load(vgg19NumpyPath, encoding='latin1').item()
        print("numpy file loaded")

    def Build(self, rgb):
        """
        load variable from numpy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = TensorFlow.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = TensorFlow.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.CreateConvolutionLayer(bgr, "conv1_1")
        self.conv1_2 = self.CreateConvolutionLayer(self.conv1_1, "conv1_2")
        self.pool1 = self.MaxPool(self.conv1_2, 'pool1')

        self.conv2_1 = self.CreateConvolutionLayer(self.pool1, "conv2_1")
        self.conv2_2 = self.CreateConvolutionLayer(self.conv2_1, "conv2_2")
        self.pool2 = self.MaxPool(self.conv2_2, 'pool2')

        self.conv3_1 = self.CreateConvolutionLayer(self.pool2, "conv3_1")
        self.conv3_2 = self.CreateConvolutionLayer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.CreateConvolutionLayer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.CreateConvolutionLayer(self.conv3_3, "conv3_4")
        self.pool3 = self.MaxPool(self.conv3_4, 'pool3')

        self.conv4_1 = self.CreateConvolutionLayer(self.pool3, "conv4_1")
        self.conv4_2 = self.CreateConvolutionLayer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.CreateConvolutionLayer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.CreateConvolutionLayer(self.conv4_3, "conv4_4")
        self.pool4 = self.MaxPool(self.conv4_4, 'pool4')

        self.conv5_1 = self.CreateConvolutionLayer(self.pool4, "conv5_1")
        self.conv5_2 = self.CreateConvolutionLayer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.CreateConvolutionLayer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.CreateConvolutionLayer(self.conv5_3, "conv5_4")
        self.pool5 = self.MaxPool(self.conv5_4, 'pool5')

        self.fc6 = self.CreateFullyConnectedLayer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = TensorFlow.nn.relu(self.fc6)

        self.fc7 = self.CreateFullyConnectedLayer(self.relu6, "fc7")
        self.relu7 = TensorFlow.nn.relu(self.fc7)

        self.fc8 = self.CreateFullyConnectedLayer(self.relu7, "fc8")

        self.prob = TensorFlow.nn.softmax(self.fc8, name="prob")

        self.dataDictionary = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def AveragePool(self, bottom, name):
        return TensorFlow.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def MaxPool(self, bottom, name):
        return TensorFlow.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def CreateConvolutionLayer(self, bottom, name):
        with TensorFlow.variable_scope(name):
            filt = self.LoadConvolutionFilters(name)

            conv = TensorFlow.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.LoadBiasWeights(name)
            bias = TensorFlow.nn.bias_add(conv, conv_biases)

            output = TensorFlow.nn.relu(bias)
            return output

    def CreateFullyConnectedLayer(self, bottom, name):
        with TensorFlow.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = TensorFlow.reshape(bottom, [-1, dim])

            weights = self.LoadFullyConnectedWeights(name)
            biases = self.LoadBiasWeights(name)

            fullyConnectedLayer = TensorFlow.nn.bias_add(TensorFlow.matmul(x, weights), biases)

            return fullyConnectedLayer

    def LoadConvolutionFilters(self, name):
        return TensorFlow.constant(self.dataDictionary[name][0], name="filter")

    def LoadBiasWeights(self, name):
        return TensorFlow.constant(self.dataDictionary[name][1], name="biases")

    def LoadFullyConnectedWeights(self, name):
        return TensorFlow.constant(self.dataDictionary[name][0], name="weights")