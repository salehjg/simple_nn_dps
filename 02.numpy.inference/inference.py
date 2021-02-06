import numpy as np
import np_model_1 as m1

path_dataset = [
    '../01.tensorflow.train/data/dataset/mnist_train_data_60000x784.npy',
    '../01.tensorflow.train/data/dataset/mnist_train_label_60000x10.npy',
    '../01.tensorflow.train/data/dataset/mnist_test_data_10000x10.npy',
    '../01.tensorflow.train/data/dataset/mnist_test_label_10000x10.npy']

path_weights = [
    '../01.tensorflow.train/data/weights/b1.npy',
    '../01.tensorflow.train/data/weights/b2.npy',
    '../01.tensorflow.train/data/weights/bout.npy',
    '../01.tensorflow.train/data/weights/h1.npy',
    '../01.tensorflow.train/data/weights/h2.npy',
    '../01.tensorflow.train/data/weights/hout.npy']

model = m1.MnistMlpInference(path_dataset, path_weights)
model.classifier_inference_run(model.dataset['test_data'], model.dataset['test_label'])
accuracy = model.classifier_inference_accuracy()
