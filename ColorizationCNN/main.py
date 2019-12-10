import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import math
from solver.layers import conv_forward_naive, conv_back_naive, relu, relu_back, max_pooling, fully_connected, \
    fully_connected_backward, max_pooling_back, batchnorm_forward, \
    batchnorm_backward
from solver.layers_fast import conv_fast, conv_fast_back
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


def polt_HSV(img):
    hsv_img = rgb2hsv(img)
    hue_img = hsv_img[:, :, 0]
    sau_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))

    ax0.imshow(img)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')
    ax3.imshow(sau_img)
    ax3.set_title("saturation channel")
    ax3.axis('off')
    plt.show()


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
        weights = {"W1": np.random.randn(3, 3, 1, 64) / np.sqrt(3276 / 2), "B1": np.zeros(64),
                   "W2": np.random.randn(3, 3, 64, 128) / np.sqrt(52428 / 2), "B2": np.zeros(128),
                   "W3": np.random.randn(3, 3, 128, 128) / np.sqrt(104857 / 2), "B3": np.zeros(128),
                   "W4": np.random.randn(3, 3, 128, 256) / np.sqrt(26214 / 2), "B4": np.zeros(256),
                   "W5": np.random.randn(3, 3, 256, 256) / np.sqrt(52428 / 2), "B5": np.zeros(256),
                   "W6": np.random.randn(3, 3, 256, 512) / np.sqrt(6400 / 2), "B6": np.zeros(512),
                   "W7": np.random.randn(3, 3, 512, 512) / np.sqrt(12800 / 2), "B7": np.zeros(512),
                   "W8": np.random.randn(3, 3, 512, 256) / np.sqrt(12800 / 2), "B8": np.zeros(256),
                   "W9": np.random.randn(1, 1, 256, 256) / np.sqrt(12800 / 2), "B9": np.zeros(256),
                   "W10": np.random.randn(3, 3, 256, 128) / np.sqrt(6400 / 2), "B10": np.zeros(128),
                   "W11": np.random.randn(3, 3, 128, 64) / np.sqrt(3200 / 2), "B11": np.zeros(64),
                   "W12": np.random.randn(3, 3, 64, 64) / np.sqrt(3200 / 2), "B12": np.zeros(64),
                   "W13": np.random.randn(3, 3, 64, 32) / np.sqrt(1600 / 2), "B13": np.zeros(32),
                   "W14": np.random.randn(3, 3, 32, 2) / np.sqrt(1600 / 2), "B14": np.zeros(2)}
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
        Z9, caches["Z9"] = conv_forward_naive(Pool8, weights["W9"], weights["B9"], {'pad': 1, 'stride': 1})
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

        # TODO: upsampling layer1 is missing here!

        # conv11
        # kernels: 128 × (3 × 3)
        # stride: 1 × 1
        Z11, caches["Z11"] = conv_forward_naive(Pool10, weights["W11"], weights["B11"], {'pad': 1, 'stride': 1})
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

        # TODO: upsampling layer2 is missing here!

        # conv13
        # kernels: 64 × (3 × 3)
        # stride: 1 × 1
        Z13, caches["Z13"] = conv_forward_naive(Pool12, weights["W13"], weights["B13"], {'pad': 1, 'stride': 1})
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

        Pool14, caches["Pool14"] = max_pooling(caches["A10"], 1)

        # TODO: upsampling layer3 is missing here!

        return Pool14, caches

    def backward_propagate(self, inputs, caches):
        x = inputs['x']
        y = inputs['y']
        gradients = {}

        dz2, gradients["W2"], gradients["B2"] = fully_connected_backward(y.reshape([-1, 2048]), caches["Z2"])
        dz2_reshape = dz2.reshape(caches["Pool1"][0].shape)
        da1 = max_pooling_back(dz2_reshape, caches["Pool1"])
        dz1 = relu_back(caches["A1"], da1)
        dbn1, gradients["gamma1"], gradients["beta1"] = batchnorm_backward(dz1, caches["BN1"])

        dz1, gradients["W1"], gradients["B1"] = conv_fast_back(dbn1, caches["Z1"])

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

    def train(self, model_inputs, lr, epochs, batch_size):

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


shape1 = x_class1.shape
num = 20
v1 = np.zeros([num, shape1[1], shape1[2], 1])
hs1 = np.zeros([num, shape1[1], shape1[2], 2])
for i in range(num):
    hsv1 = rgb2hsv(x_class1[i, :, :, :])
    v1[i, :, :, 0] = hsv1[:, :, 0]
    hs1[i, :, :, :] = hsv1[:, :, 1:num]

input1 = {"x": v1, "y": hs1}
cnn1 = ColorizationCNN(input1, v1, hs1)
cnn1.train(input1, 0.005, 2, 5)
# output1, caches1 = cnn1.forward_propagate(input1, w1, param1, bn_param1)
# gradients1 = cnn1.backward_propagate(output1, caches1)
# print(hs1.reshape([2, -1]).shape == output1.shape)
