import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
import math
import cv2
from solver.layers import conv_forward_naive, conv_back_naive, relu, relu_back, max_pooling, fully_connected, \
    fully_connected_backward, max_pooling_back, batchnorm_forward, deconv_forward, deconv_backward, mean_equared_error, \
    batchnorm_backward, mean_equared_error_back
from solver.layers_fast import conv_fast, conv_fast_back
import os
from terminaltables import AsciiTable
from cifar_loader import cifar10

cifar10.maybe_download_and_extract()
train_x_raw, train_y, train_ = cifar10.load_training_data()
test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
classes = cifar10.load_class_names()

# select one class
index_of_class1 = np.where(train_y == 1)[0]
x_class1 = train_x_raw[index_of_class1, :]
y_class1 = train_y[index_of_class1]
y_one_hot_class1 = train_[index_of_class1]


class ColorizationCNN:
    _weights = {}
    _params = {}
    _bn_params = {}

    _rms_velocity = {}
    _momentum_velocity = {}

    def __init__(self, model_inputs, x, y):
        model_inputs["x"] = x
        model_inputs["y"] = y
        self._weights = self.init_weights()
        self._params = self.init_params()
        self._bn_params = self.init_bn_params()

    def init_weights(self):
        # W.shape: (window of filter, window of filter, channel of the previous layer, channel of this layer)
        # W = np.random.randn(n) / sqrt(n)  calibrating the variances with 1/sqrt(n)
        weights = {"W1": np.random.randn(3, 3, 1, 64) / np.sqrt(32 / 2), "B1": np.zeros(64),
                   "W2": np.random.randn(3, 3, 64, 128) / np.sqrt(524 / 2), "B2": np.zeros(128),
                   "W3": np.random.randn(3, 3, 128, 128) / np.sqrt(104 / 2), "B3": np.zeros(128),
                   "W4": np.random.randn(3, 3, 128, 256) / np.sqrt(262 / 2), "B4": np.zeros(256),
                   "W5": np.random.randn(3, 3, 256, 256) / np.sqrt(524 / 2), "B5": np.zeros(256),
                   "W6": np.random.randn(3, 3, 256, 512) / np.sqrt(64 / 2), "B6": np.zeros(512),
                   "W7": np.random.randn(3, 3, 512, 512) / np.sqrt(128 / 2), "B7": np.zeros(512),
                   "W8": np.random.randn(3, 3, 512, 256) / np.sqrt(128 / 2), "B8": np.zeros(256),
                   "W9": np.random.randn(1, 1, 256, 256) / np.sqrt(128 / 2), "B9": np.zeros(256),
                   "W10": np.random.randn(3, 3, 256, 128) / np.sqrt(64 / 2), "B10": np.zeros(128),
                   "USW1": np.random.randn(3, 3, 128, 64) / np.sqrt(8 / 2), "USB1": np.zeros(64),
                   "W11": np.random.randn(3, 3, 64, 64) / np.sqrt(320 / 2), "B11": np.zeros(64),
                   "W12": np.random.randn(3, 3, 64, 64) / np.sqrt(32 / 2), "B12": np.zeros(64),
                   "USW2": np.random.randn(3, 3, 64, 32) / np.sqrt(8 / 2), "USB2": np.zeros(32),
                   "W13": np.random.randn(3, 3, 32, 32) / np.sqrt(16 / 2), "B13": np.zeros(32),
                   "W14": np.random.randn(3, 3, 32, 2) / np.sqrt(16 / 2), "B14": np.zeros(2),
                   "USW3": np.random.randn(3, 3, 2, 2) / np.sqrt(8 / 2), "USB3": np.zeros(2)}
        # Init adam running means
        for key in weights:
            self._rms_velocity[key] = 0
            self._momentum_velocity[key] = 0
        return weights

    def init_params(self):
        params = {"gamma1": np.ones(64), "beta1": np.zeros(64),
                  "gamma2": np.ones(128), "beta2": np.zeros(128),
                  "gamma3": np.ones(128), "beta3": np.zeros(128),
                  "gamma4": np.ones(256), "beta4": np.zeros(256),
                  "gamma5": np.ones(256), "beta5": np.zeros(256),
                  "gamma6": np.ones(512), "beta6": np.zeros(512),
                  "gamma7": np.ones(512), "beta7": np.zeros(512),
                  "gamma8": np.ones(256), "beta8": np.zeros(256),
                  "gamma9": np.ones(256), "beta9": np.zeros(256),
                  "gamma10": np.ones(128), "beta10": np.zeros(128),
                  "gamma11": np.ones(64), "beta11": np.zeros(64),
                  "gamma12": np.ones(64), "beta12": np.zeros(64),
                  "gamma13": np.ones(32), "beta13": np.zeros(32),
                  "gamma14": np.ones(2), "beta14": np.zeros(2)}

        return params

    def init_bn_params(self):
        bn_params = {"running_mu_1": np.zeros(64), "running_sigma_1": np.zeros(64),
                     "running_mu_2": np.zeros(128), "running_sigma_2": np.zeros(128),
                     "running_mu_3": np.zeros(128), "running_sigma_3": np.zeros(128),
                     "running_mu_4": np.zeros(256), "running_sigma_4": np.zeros(256),
                     "running_mu_5": np.zeros(256), "running_sigma_5": np.zeros(256),
                     "running_mu_6": np.zeros(512), "running_sigma_6": np.zeros(512),
                     "running_mu_7": np.zeros(512), "running_sigma_7": np.zeros(512),
                     "running_mu_8": np.zeros(256), "running_sigma_8": np.zeros(256),
                     "running_mu_9": np.zeros(256), "running_sigma_9": np.zeros(256),
                     "running_mu_10": np.zeros(128), "running_sigma_10": np.zeros(128),
                     "running_mu_11": np.zeros(64), "running_sigma_11": np.zeros(64),
                     "running_mu_12": np.zeros(64), "running_sigma_12": np.zeros(64),
                     "running_mu_13": np.zeros(32), "running_sigma_13": np.zeros(32),
                     "running_mu_14": np.zeros(2), "running_sigma_14": np.zeros(2)}

        return bn_params

    def save_model(self, filename):
        model = {}

        for weight in self._weights:
            model[weight] = self._weights[weight]

        for param in self._params:
            model[param] = self._params[param]

        for bn in self._bn_params:
            model[bn] = self._bn_params[bn]

        np.save(filename + '.npy', model)
        print("Successfully saved weights")

    def load_model_from_file(self, filename):
        weights = {}
        params = {}
        bn_params = {}

        data_load = np.load(filename + '.npy', allow_pickle=True).item()

        weights["W1"] = data_load.get('W1')
        weights["W2"] = data_load.get('W2')
        weights["W3"] = data_load.get('W3')
        weights["W4"] = data_load.get('W4')
        weights["W5"] = data_load.get('W5')
        weights["W6"] = data_load.get('W6')
        weights["W7"] = data_load.get('W7')
        weights["W8"] = data_load.get('W8')
        weights["W9"] = data_load.get('W9')
        weights["W10"] = data_load.get('W10')
        weights["W11"] = data_load.get('W11')
        weights["W12"] = data_load.get('W12')
        weights["W13"] = data_load.get('W13')
        weights["W14"] = data_load.get('W14')

        weights["B1"] = data_load.get('B1')
        weights["B2"] = data_load.get('B2')
        weights["B3"] = data_load.get('B3')
        weights["B4"] = data_load.get('B4')
        weights["B5"] = data_load.get('B5')
        weights["B6"] = data_load.get('B6')
        weights["B7"] = data_load.get('B7')
        weights["B8"] = data_load.get('B8')
        weights["B9"] = data_load.get('B9')
        weights["B10"] = data_load.get('B10')
        weights["B11"] = data_load.get('B11')
        weights["B12"] = data_load.get('B12')
        weights["B13"] = data_load.get('B13')
        weights["B14"] = data_load.get('B14')

        weights["USW1"] = data_load.get('USW1')
        weights["USW2"] = data_load.get('USW2')
        weights["USW3"] = data_load.get('USW3')
        weights["USB1"] = data_load.get('USB1')
        weights["USB2"] = data_load.get('USB2')
        weights["USB3"] = data_load.get('USB3')

        params["gamma1"] = data_load.get('gamma1')
        params["beta1"] = data_load.get('beta1')
        params["gamma2"] = data_load.get('gamma2')
        params["beta2"] = data_load.get('beta2')
        params["gamma3"] = data_load.get('gamma3')
        params["beta3"] = data_load.get('beta3')
        params["gamma3"] = data_load.get('gamma3')
        params["beta3"] = data_load.get('beta3')
        params["gamma4"] = data_load.get('gamma4')
        params["beta4"] = data_load.get('beta4')
        params["gamma5"] = data_load.get('gamma5')
        params["beta5"] = data_load.get('beta5')
        params["gamma6"] = data_load.get('gamma6')
        params["beta6"] = data_load.get('beta6')
        params["gamma7"] = data_load.get('gamma7')
        params["beta7"] = data_load.get('beta7')
        params["gamma8"] = data_load.get('gamma8')
        params["beta8"] = data_load.get('beta8')
        params["gamma9"] = data_load.get('gamma9')
        params["beta9"] = data_load.get('beta9')
        params["gamma10"] = data_load.get('gamma10')
        params["beta10"] = data_load.get('beta10')
        params["gamma11"] = data_load.get('gamma11')
        params["beta11"] = data_load.get('beta11')
        params["gamma12"] = data_load.get('gamma12')
        params["beta12"] = data_load.get('beta12')
        params["gamma13"] = data_load.get('gamma13')
        params["beta13"] = data_load.get('beta13')
        params["gamma14"] = data_load.get('gamma14')
        params["beta14"] = data_load.get('beta14')

        bn_params["running_mu_1"] = data_load.get('running_mu_1')
        bn_params["running_sigma_1"] = data_load.get('running_sigma_1')
        bn_params["running_mu_2"] = data_load.get('running_mu_2')
        bn_params["running_sigma_2"] = data_load.get('running_sigma_2')
        bn_params["running_mu_3"] = data_load.get('running_mu_3')
        bn_params["running_sigma_3"] = data_load.get('running_sigma_3')
        bn_params["running_mu_4"] = data_load.get('running_mu_4')
        bn_params["running_sigma_4"] = data_load.get('running_sigma_4')
        bn_params["running_mu_5"] = data_load.get('running_mu_5')
        bn_params["running_sigma_5"] = data_load.get('running_sigma_5')
        bn_params["running_mu_6"] = data_load.get('running_mu_6')
        bn_params["running_sigma_6"] = data_load.get('running_sigma_6')
        bn_params["running_mu_7"] = data_load.get('running_mu_7')
        bn_params["running_sigma_7"] = data_load.get('running_sigma_7')
        bn_params["running_mu_8"] = data_load.get('running_mu_8')
        bn_params["running_sigma_8"] = data_load.get('running_sigma_8')
        bn_params["running_mu_9"] = data_load.get('running_mu_9')
        bn_params["running_sigma_9"] = data_load.get('running_sigma_9')
        bn_params["running_mu_10"] = data_load.get('running_mu_10')
        bn_params["running_sigma_10"] = data_load.get('running_sigma_10')
        bn_params["running_mu_11"] = data_load.get('running_mu_11')
        bn_params["running_sigma_11"] = data_load.get('running_sigma_11')
        bn_params["running_mu_12"] = data_load.get('running_mu_12')
        bn_params["running_sigma_12"] = data_load.get('running_sigma_12')
        bn_params["running_mu_13"] = data_load.get('running_mu_13')
        bn_params["running_sigma_13"] = data_load.get('running_sigma_13')
        bn_params["running_mu_14"] = data_load.get('running_mu_14')
        bn_params["running_sigma_14"] = data_load.get('running_sigma_14')
        for key in weights:
            self._rms_velocity[key] = 0
            self._momentum_velocity[key] = 0

        return weights, params, bn_params

    def forward_propagate(self, model_inputs, weights, params, bn_params, run='train'):
        x = model_inputs["x"]
        y = model_inputs["y"]

        caches = {}
        """
        Encoder
        """
        # conv1
        # kernels: 64 × (3 × 3)
        # stride: 2 × 2
        Z1, caches["Z1"] = conv_forward_naive(x, weights["W1"], weights["B1"], {'pad': 1, 'stride': 2})
        BN1, bn_params["running_mu_1"], bn_params["running_sigma_1"], caches["BN1"] = batchnorm_forward(Z1, params[
            "gamma1"], params["beta1"], bn_params["running_mu_1"], bn_params["running_sigma_1"], run)

        caches["A1"] = relu(BN1)

        Pool1, caches["Pool1"] = max_pooling(caches["A1"], 1)

        # conv2
        # kernels: 128 × (3 × 3)
        # stride: 1 × 1
        Z2, caches["Z2"] = conv_forward_naive(Pool1, weights["W2"], weights["B2"], {'pad': 1, 'stride': 1})

        BN2, bn_params["running_mu_2"], bn_params["running_sigma_2"], caches["BN2"] = batchnorm_forward(Z2, params[
            "gamma2"], params["beta2"], bn_params["running_mu_2"], bn_params["running_sigma_2"], run)

        caches["A2"] = relu(BN2)

        Pool2, caches["Pool2"] = max_pooling(caches["A2"], 1)

        # conv3
        # kernels: 128 × (3 × 3)
        # stride: 2 × 2
        Z3, caches["Z3"] = conv_forward_naive(Pool2, weights["W3"], weights["B3"], {'pad': 1, 'stride': 2})
        BN3, bn_params["running_mu_3"], bn_params["running_sigma_3"], caches["BN3"] = batchnorm_forward(Z3, params[
            "gamma3"], params["beta3"], bn_params["running_mu_3"], bn_params["running_sigma_3"], run)

        caches["A3"] = relu(BN3)

        Pool3, caches["Pool3"] = max_pooling(caches["A3"], 1)
        # conv4
        # kernels: 256 × (3 × 3)
        # stride: 1 × 1
        Z4, caches["Z4"] = conv_forward_naive(Pool3, weights["W4"], weights["B4"], {'pad': 1, 'stride': 1})
        BN4, bn_params["running_mu_4"], bn_params["running_sigma_4"], caches["BN4"] = batchnorm_forward(Z4, params[
            "gamma4"], params["beta4"], bn_params["running_mu_4"], bn_params["running_sigma_4"], run)

        caches["A4"] = relu(BN4)

        Pool4, caches["Pool4"] = max_pooling(caches["A4"], 1)

        # conv5
        # kernels: 256 × (3 × 3)
        # stride: 2 × 2
        Z5, caches["Z5"] = conv_forward_naive(Pool4, weights["W5"], weights["B5"], {'pad': 1, 'stride': 2})
        BN5, bn_params["running_mu_5"], bn_params["running_sigma_5"], caches["BN5"] = batchnorm_forward(Z5, params[
            "gamma5"], params["beta5"], bn_params["running_mu_5"], bn_params["running_sigma_5"], run)

        caches["A5"] = relu(BN5)

        Pool5, caches["Pool5"] = max_pooling(caches["A5"], 1)
        # conv6
        # kernels: 512 × (3 × 3)
        # stride: 1 × 1
        Z6, caches["Z6"] = conv_forward_naive(Pool5, weights["W6"], weights["B6"], {'pad': 1, 'stride': 1})

        BN6, bn_params["running_mu_6"], bn_params["running_sigma_6"], caches["BN6"] = batchnorm_forward(Z6, params[
            "gamma6"], params["beta6"], bn_params["running_mu_6"], bn_params["running_sigma_6"], run)

        caches["A6"] = relu(BN6)

        Pool6, caches["Pool6"] = max_pooling(caches["A6"], 1)

        # conv7
        # kernels: 512 × (3 × 3)
        # stride: 1 × 1
        Z7, caches["Z7"] = conv_forward_naive(Pool6, weights["W7"], weights["B7"], {'pad': 1, 'stride': 1})
        BN7, bn_params["running_mu_7"], bn_params["running_sigma_7"], caches["BN7"] = batchnorm_forward(Z7, params[
            "gamma7"], params["beta7"], bn_params["running_mu_7"], bn_params["running_sigma_7"], run)

        caches["A7"] = relu(BN7)

        Pool7, caches["Pool7"] = max_pooling(caches["A7"], 1)

        # conv8
        # kernels: 256 × (3 × 3)
        # stride: 1 × 1
        Z8, caches["Z8"] = conv_forward_naive(Pool7, weights["W8"], weights["B8"], {'pad': 1, 'stride': 1})

        BN8, bn_params["running_mu_8"], bn_params["running_sigma_8"], caches["BN8"] = batchnorm_forward(Z8, params[
            "gamma8"], params["beta8"], bn_params["running_mu_8"], bn_params["running_sigma_8"], run)

        caches["A8"] = relu(BN8)

        Pool8, caches["Pool8"] = max_pooling(caches["A8"], 1)
        """
        Fusion
        """
        # TODO: fusion is missing! Need to use inception-ResNet-v2
        # conv9
        # kernels: 256 × (1 × 1)
        # stride: 1 × 1
        Z9, caches["Z9"] = conv_forward_naive(Pool8, weights["W9"], weights["B9"], {'pad': 0, 'stride': 1})

        BN9, bn_params["running_mu_9"], bn_params["running_sigma_9"], caches["BN9"] = batchnorm_forward(Z9, params[
            "gamma9"], params["beta9"], bn_params["running_mu_9"], bn_params["running_sigma_9"], run)

        caches["A9"] = relu(BN9)

        Pool9, caches["Pool9"] = max_pooling(caches["A9"], 1)
        """
        Decoder
        """
        # conv10
        # kernels: 256 × (1 × 1)
        # stride: 1 × 1
        Z10, caches["Z10"] = conv_forward_naive(Pool9, weights["W10"], weights["B10"], {'pad': 1, 'stride': 1})
        BN10, bn_params["running_mu_10"], bn_params["running_sigma_10"], caches["BN10"] = batchnorm_forward(Z10, params[
            "gamma10"], params["beta10"], bn_params["running_mu_10"], bn_params["running_sigma_10"], run)

        caches["A10"] = relu(BN10)

        Pool10, caches["Pool10"] = max_pooling(caches["A10"], 1)

        # upsampling layer1
        US1, caches["US1"] = deconv_forward(Pool10, weights["USW1"], weights["USB1"])

        # conv11
        # kernels: 128 × (3 × 3)
        # stride: 1 × 1
        Z11, caches["Z11"] = conv_forward_naive(US1, weights["W11"], weights["B11"], {'pad': 1, 'stride': 1})
        BN11, bn_params["running_mu_11"], bn_params["running_sigma_11"], caches["BN11"] = batchnorm_forward(Z11, params[
            "gamma11"], params["beta11"], bn_params["running_mu_11"], bn_params["running_sigma_11"], run)

        caches["A11"] = relu(BN11)

        Pool11, caches["Pool11"] = max_pooling(caches["A11"], 1)

        # conv12
        # kernels: 64 × (3 × 3)
        # stride: 1 × 1
        Z12, caches["Z12"] = conv_forward_naive(Pool11, weights["W12"], weights["B12"], {'pad': 1, 'stride': 1})
        BN12, bn_params["running_mu_12"], bn_params["running_sigma_12"], caches["BN12"] = batchnorm_forward(Z12, params[
            "gamma12"], params["beta12"], bn_params["running_mu_12"], bn_params["running_sigma_12"], run)

        caches["A12"] = relu(BN12)

        Pool12, caches["Pool12"] = max_pooling(caches["A12"], 1)

        # upsampling layer2
        US2, caches["US2"] = deconv_forward(Pool12, weights["USW2"], weights["USB2"])

        # conv13
        # kernels: 64 × (3 × 3)
        # stride: 1 × 1
        Z13, caches["Z13"] = conv_forward_naive(US2, weights["W13"], weights["B13"], {'pad': 1, 'stride': 1})
        BN13, bn_params["running_mu_13"], bn_params["running_sigma_13"], caches["BN13"] = batchnorm_forward(Z13, params[
            "gamma13"], params["beta13"], bn_params["running_mu_13"], bn_params["running_sigma_13"], run)

        caches["A13"] = relu(BN13)

        Pool13, caches["Pool13"] = max_pooling(caches["A13"], 1)

        # conv14
        # kernels: 2 × (3 × 3)
        # stride: 1 × 1
        Z14, caches["Z14"] = conv_forward_naive(Pool13, weights["W14"], weights["B14"], {'pad': 1, 'stride': 1})

        BN14, bn_params["running_mu_14"], bn_params["running_sigma_14"], caches["BN14"] = batchnorm_forward(Z14, params[
            "gamma14"], params["beta14"], bn_params["running_mu_14"], bn_params["running_sigma_14"], run)

        caches["A14"] = relu(BN14)

        Pool14, caches["Pool14"] = max_pooling(caches["A14"], 1)

        # upsampling layer3
        US3, caches["US3"] = deconv_forward(Pool14, weights["USW3"], weights["USB3"])

        loss = mean_equared_error(US3, y)
        caches["HS"] = US3

        return loss, caches

    def backward_propagate(self, inputs, caches):
        x = inputs['x']
        y = inputs['y']
        gradients = {}

        dl = mean_equared_error_back(caches["HS"], y)

        du3, gradients["USW3"], gradients["USB3"] = deconv_backward(dl, caches["US3"])

        da14 = max_pooling_back(du3, caches["Pool14"])
        dz14 = relu_back(caches["A14"], da14)
        dbn14, gradients["gamma14"], gradients["beta14"] = batchnorm_backward(dz14, caches["BN14"])
        dz14, gradients["W14"], gradients["B14"] = conv_back_naive(dbn14, caches["Z14"])

        da13 = max_pooling_back(dz14, caches["Pool13"])
        dz13 = relu_back(caches["A13"], da13)
        dbn13, gradients["gamma13"], gradients["beta13"] = batchnorm_backward(dz13, caches["BN13"])
        dz13, gradients["W13"], gradients["B13"] = conv_back_naive(dbn13, caches["Z13"])

        du2, gradients["USW2"], gradients["USB2"] = deconv_backward(dz13, caches["US2"])

        da12 = max_pooling_back(du2, caches["Pool12"])
        dz12 = relu_back(caches["A12"], da12)
        dbn12, gradients["gamma12"], gradients["beta12"] = batchnorm_backward(dz12, caches["BN12"])
        dz12, gradients["W12"], gradients["B12"] = conv_back_naive(dbn12, caches["Z12"])

        da11 = max_pooling_back(dz12, caches["Pool11"])
        dz11 = relu_back(caches["A11"], da11)
        dbn11, gradients["gamma11"], gradients["beta11"] = batchnorm_backward(dz11, caches["BN11"])
        dz11, gradients["W11"], gradients["B11"] = conv_back_naive(dbn11, caches["Z11"])

        du1, gradients["USW1"], gradients["USB1"] = deconv_backward(dz11, caches["US1"])

        da10 = max_pooling_back(du1, caches["Pool10"])
        dz10 = relu_back(caches["A10"], da10)
        dbn10, gradients["gamma10"], gradients["beta10"] = batchnorm_backward(dz10, caches["BN10"])
        dz10, gradients["W10"], gradients["B10"] = conv_back_naive(dbn10, caches["Z10"])

        da9 = max_pooling_back(dz10, caches["Pool9"])
        dz9 = relu_back(caches["A9"], da9)
        dbn9, gradients["gamma9"], gradients["beta9"] = batchnorm_backward(dz9, caches["BN9"])
        dz9, gradients["W9"], gradients["B9"] = conv_back_naive(dbn9, caches["Z9"])

        da8 = max_pooling_back(dz9, caches["Pool8"])
        dz8 = relu_back(caches["A8"], da8)
        dbn8, gradients["gamma8"], gradients["beta8"] = batchnorm_backward(dz8, caches["BN8"])
        dz8, gradients["W8"], gradients["B8"] = conv_back_naive(dbn8, caches["Z8"])

        da7 = max_pooling_back(dz8, caches["Pool7"])
        dz7 = relu_back(caches["A7"], da7)
        dbn7, gradients["gamma7"], gradients["beta7"] = batchnorm_backward(dz7, caches["BN7"])
        dz7, gradients["W7"], gradients["B7"] = conv_back_naive(dbn7, caches["Z7"])

        da6 = max_pooling_back(dz7, caches["Pool6"])
        dz6 = relu_back(caches["A6"], da6)
        dbn6, gradients["gamma6"], gradients["beta6"] = batchnorm_backward(dz6, caches["BN6"])
        dz6, gradients["W6"], gradients["B6"] = conv_back_naive(dbn6, caches["Z6"])

        da5 = max_pooling_back(dz6, caches["Pool5"])
        dz5 = relu_back(caches["A5"], da5)
        dbn5, gradients["gamma5"], gradients["beta5"] = batchnorm_backward(dz5, caches["BN5"])
        dz5, gradients["W5"], gradients["B5"] = conv_back_naive(dbn5, caches["Z5"])

        da4 = max_pooling_back(dz5, caches["Pool4"])
        dz4 = relu_back(caches["A4"], da4)
        dbn4, gradients["gamma4"], gradients["beta4"] = batchnorm_backward(dz4, caches["BN4"])
        dz4, gradients["W4"], gradients["B4"] = conv_back_naive(dbn4, caches["Z4"])

        da3 = max_pooling_back(dz4, caches["Pool3"])
        dz3 = relu_back(caches["A3"], da3)
        dbn3, gradients["gamma3"], gradients["beta3"] = batchnorm_backward(dz3, caches["BN3"])
        dz3, gradients["W3"], gradients["B3"] = conv_back_naive(dbn3, caches["Z3"])

        da2 = max_pooling_back(dz3, caches["Pool2"])
        dz2 = relu_back(caches["A2"], da2)
        dbn2, gradients["gamma2"], gradients["beta2"] = batchnorm_backward(dz2, caches["BN2"])
        dz2, gradients["W2"], gradients["B2"] = conv_back_naive(dbn2, caches["Z2"])

        da1 = max_pooling_back(dz2, caches["Pool1"])
        dz1 = relu_back(caches["A1"], da1)
        dbn1, gradients["gamma1"], gradients["beta1"] = batchnorm_backward(dz1, caches["BN1"])
        dz1, gradients["W1"], gradients["B1"] = conv_back_naive(dbn1, caches["Z1"])

        return gradients

    def update_adam(self, gradients, iteration, lr):
        beta = 0.9
        # repeat for each weight
        for key in self._weights:
            self._rms_velocity[key] = beta * self._rms_velocity[key] + ((1 - beta) * np.square(gradients[key]))
            rms_corrected = self._rms_velocity[key] / (1 - math.pow(beta, iteration))
            self._momentum_velocity[key] = beta * self._momentum_velocity[key] + ((1 - beta) * gradients[key])
            momentum_corrected = self._momentum_velocity[key] / (1 - math.pow(beta, iteration))
            self._weights[key] -= lr * momentum_corrected / np.sqrt(rms_corrected + 1e-8)

        for key in self._params:
            self._params[key] -= lr * gradients[key]

    def train(self, model_inputs, lr, epochs, batch_size, print_every):

        run_id = str(np.random.randint(1000))

        print("\nTraining with run id:" + run_id)

        print("lr: " + str(lr))
        print("epochs: " + str(epochs))
        print("batch size: " + str(batch_size))

        # we need method called update weights

        epoch_size = int(model_inputs["x"].shape[0] / batch_size)
        inputs_x = model_inputs["x"]
        inputs_y = model_inputs["y"]

        shuffle_index = np.arange(inputs_x.shape[0])
        it = 0
        for e in range(epochs):
            # shuffle data
            shuffle_index = np.arange(inputs_x.shape[0])
            np.random.shuffle(shuffle_index)

            inputs_x = inputs_x[shuffle_index, ...]
            inputs_y = inputs_y[shuffle_index, ...]

            for b in range(epoch_size):
                batch_x = inputs_x[batch_size * b:batch_size * (b + 1), ...]
                batch_y = inputs_y[batch_size * b:batch_size * (b + 1), ...]

                batch_inputs = {"x": batch_x, "y": batch_y}
                cost, caches = self.forward_propagate(batch_inputs, self._weights, self._params, self._bn_params)
                gradients = self.backward_propagate(batch_inputs, caches)
                self.update_adam(gradients, i + 1, lr)

                if it % print_every == 0:
                    data = [["Progress", "MSE Value"],
                            [str(int(b / float(epoch_size) * 100)) + "% " + str(e) + "/" + str(epochs), cost]]

                    table = AsciiTable(data)
                    table.title = "Stats run_" + run_id

                    os.system('clear')
                    print(table.table)
                    print("Printing every " + str(print_every) + " iterations")

                it += 1

    def evaluate(self, img):
        shape = img.shape
        l = np.zeros([1, shape[0], shape[1], 1])
        ab_true = np.zeros([1, shape[0], shape[1], 2])
        lab = color.rgb2lab(img)
        l[0, :, :, 0] = lab[:, :, 0]
        ab_true[0, :, :, :] = lab[:, :, 1:2]
        lab_input = {"x": l, "y": ab_true}
        _, cache = self.forward_propagate(lab_input, self._weights, self._params, self._bn_params)
        ab = cache["HS"]
        colorized_img = np.zeros([32, 32, 3])
        colorized_img[:, :, 0] = l[0, :, :, 0]
        colorized_img[:, :, 1] = ab[0, :, :, 0]
        colorized_img[:, :, 2] = ab[0, :, :, 1]
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(4, 4))
        ax0.imshow(img)
        ax0.set_title("original image")
        ax0.axis('off')
        ax2.imshow(color.gray2rgb(color.rgb2gray(img)))
        ax2.set_title("gray-scale image")
        ax2.axis('off')
        ax1.imshow(color.lab2rgb(colorized_img))
        ax1.set_title("colorized image")
        ax1.axis('off')
        plt.show()


shape1 = x_class1.shape
# the amount of training data
num = 1
l1 = np.zeros([num, shape1[1], shape1[2], 1])
ab1 = np.zeros([num, shape1[1], shape1[2], 2])
for i in range(num):
    lab1 = color.rgb2lab(x_class1[i, :, :, :])
    l1[i, :, :, 0] = lab1[:, :, 0]
    ab1[i, :, :, 0] = lab1[:, :, 1]
    ab1[i, :, :, 1] = lab1[:, :, 2]

input1 = {"x": l1, "y": ab1}

cnn1 = ColorizationCNN(input1, l1, ab1)
cnn1.load_model_from_file("/Users/coomaqin/PycharmProjects/intro_to_Ai/intro_to_Ai/ColorizationCNN/weights_2")
cnn1.evaluate(x_class1[2000])
# cnn1.train(input1, 0.001, 2, 5, 2)


# W1 = np.random.randn(3, 3, 1, 64) / np.sqrt(3276 / 2)
# B1 = np.zeros(64)
# print(input1["x"].shape)
# output1, caches1 = deconv_forward(input1["x"], W1, B1)
# print(output1.shape)
# dx, dw, db = deconv_backward(output1, caches1)
# print(dx.shape)
