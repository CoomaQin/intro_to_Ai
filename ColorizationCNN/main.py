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
        weights = {"W1": np.random.randn(1, 1, 1, 16) / np.sqrt(16384 / 2), "B1": np.zeros(16),
                   "W2": np.random.randn(4624, 2048) / np.sqrt(4624 / 2), "B2": np.zeros(2048)}
        # Init adam running means
        for key in weights:
            self._rms_velocity[key] = 0
            self._momentum_velocity[key] = 0
        return weights

    def init_params(self):
        params = {"gamma1": np.ones(16), "beta1": np.zeros(16)}

        return params

    def init_bn_params(self):
        bn_params = {"running_mu_1": np.zeros(16), "running_sigma_1": np.zeros(16)}

        return bn_params

    def forward_propagate(self, model_inputs, weights, params, bn_params, run='train'):
        x = model_inputs["x"]
        y = model_inputs["y"]

        caches = {}

        Z1, caches["Z1"] = conv_fast(x, weights["W1"], weights["B1"], {'pad': 1, 'stride': 1})

        BN1, bn_params["running_mu_1"], bn_params["running_sigma_1"], caches["BN1"] = batchnorm_forward(Z1, params[
            "gamma1"], params["beta1"], bn_params["running_mu_1"], bn_params["running_sigma_1"], run)

        caches["A1"] = relu(BN1)

        Pool1, caches["Pool1"] = max_pooling(caches["A1"], 2)

        pool1_reshape = Pool1.reshape(Pool1.shape[0], Pool1.shape[1] * Pool1.shape[2] * Pool1.shape[3])

        Z2, caches["Z2"] = fully_connected(pool1_reshape, weights["W2"], weights["B2"])

        fc_out = Z2
        # caches["A2"] = softmax(Z2)
        # cost = np.mean(softmax_cost(y, caches["A2"]))

        return fc_out, caches

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
