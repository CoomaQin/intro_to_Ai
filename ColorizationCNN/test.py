from cifar_loader import cifar10
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from solver.layers import conv_forward_naive, conv_back_naive, relu, relu_back, max_pooling, fully_connected, \
    fully_connected_backward, softmax, softmax_cost, softmax_back, max_pooling_back, batchnorm_forward, \
    batchnorm_backward
from solver.layers_fast import conv_fast, conv_fast_back

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

    def __init__(self, model_inputs, x, y):
        model_inputs["x"] = x
        model_inputs["y"] = y

    def init_weights(self):
        weights = {"W1": np.random.randn(1, 1, 1, 16) / np.sqrt(16384 / 2), "B1": np.zeros(16),
                   "W2": np.random.randn(4624, 2048) / np.sqrt(4624 / 2), "B2": np.zeros(2048)}
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


        cost = Z2
        # caches["A2"] = softmax(Z2)
        # cost = np.mean(softmax_cost(y, caches["A2"]))

        return cost, caches


shape1 = x_class1.shape
v1 = np.zeros([2, shape1[1], shape1[2], 1])
hs1 = np.zeros([2, shape1[1], shape1[2], 2])
for i in range(2):
    hsv1 = rgb2hsv(x_class1[i, :, :, :])
    v1[i, :, :, 0] = hsv1[:, :, 0]
    hs1[i, :, :, :] = hsv1[:, :, 1:2]

input1 = {}
cnn1 = ColorizationCNN(input1, v1, hs1)
w1 = cnn1.init_weights()
param1 = cnn1.init_params()
bn_param1 = cnn1.init_bn_params()
output1 = cnn1.forward_propagate(input1, w1, param1, bn_param1)
print(output1[0].shape)