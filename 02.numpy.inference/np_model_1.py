import numpy as np


class MnistMlpInference:
    def __init__(self, path_dataset, path_weights):
        self.weights = {}
        self.dataset = {}
        try:
            self.dataset['train_data'] = np.load(path_dataset[0])
            self.dataset['train_label'] = np.load(path_dataset[1])
            self.dataset['test_data'] = np.load(path_dataset[2])
            self.dataset['test_label'] = np.load(path_dataset[3])

            self.weights['b1'] = np.load(path_weights[0])
            self.weights['b2'] = np.load(path_weights[1])
            self.weights['bout'] = np.load(path_weights[2])
            self.weights['h1'] = np.load(path_weights[3])
            self.weights['h2'] = np.load(path_weights[4])
            self.weights['hout'] = np.load(path_weights[5])
        except:
            print('Errors have occurred trying to load the weights and/or the dataset.')

    def classifier_inference_run(self, inputs, labels):
        # convert labels to one-hot encoding
        def batch_make_onehot(batch_labels, nclasses):
            rslts = np.zeros(shape=[batch_labels.shape[0], nclasses])
            for i in range(batch_labels.shape[0]):
                rslts[i, :] = make_onehot(batch_labels[i], nclasses)
            return rslts

        def make_onehot(label, nclasses):
            return np.eye(nclasses, dtype=np.int32)[label]

        def multilayer_perceptron(x):
            layer_1 = np.add(np.matmul(x, self.weights['h1']), self.weights['b1'])
            layer_2 = np.add(np.matmul(layer_1, self.weights['h2']), self.weights['b2'])
            out_layer = np.matmul(layer_2, self.weights['hout']) + self.weights['bout']
            return out_layer
        logits = multilayer_perceptron(inputs)

        a1 = np.argmax(logits, axis=1)
        a2 = np.argmax(labels, axis=1)
        correct = np.equal(a1, a2)

        self.accuracy = np.sum(correct.astype(dtype=np.float32)) / float(inputs.shape[0])
        print('Accuracy: ', self.accuracy)

    def classifier_inference_accuracy(self):
        return self.accuracy