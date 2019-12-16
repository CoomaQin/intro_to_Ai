import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
from cifar_loader import cifar10
from solver.solver import ColorizationCNN

cifar10.maybe_download_and_extract()
train_x_raw, train_y, train_ = cifar10.load_training_data()
test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
classes = cifar10.load_class_names()

# select one class
index_of_class1 = np.where(train_y == 1)[0]
x_class1 = train_x_raw[index_of_class1, :]
y_class1 = train_y[index_of_class1]
y_one_hot_class1 = train_[index_of_class1]


shape1 = x_class1.shape
# the amount of training data
num = 80
l1 = np.zeros([num, shape1[1], shape1[2], 1])
ab1 = np.zeros([num, shape1[1], shape1[2], 2])
for i in range(num):
    lab1 = color.rgb2lab(x_class1[i, :, :, :])
    l1[i, :, :, 0] = lab1[:, :, 0]
    ab1[i, :, :, 0] = lab1[:, :, 1]
    ab1[i, :, :, 1] = lab1[:, :, 2]

input1 = {"x": l1, "y": ab1}
cnn1 = ColorizationCNN(input1, l1, ab1)
cnn1.load_model_from_file("/Users/Coomaqin/PycharmProjects/intro_to_Ai/intro_to_Ai/ColorizationCNN/weights_2")
# cnn1.train(input1, 0.02, 3, 20, 2)
# cnn1.save_model("/Users/Andrey/PycharmProjects/intro_to_Ai/intro_to_Ai/ColorizationCNN/weights_13")
cnn1.evaluate(x_class1[1])
# cnn1.evaluate(x_class1[2])
# cnn1.evaluate(x_class1[55])
