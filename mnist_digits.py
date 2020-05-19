import read_idx
import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayerNN(nn.Module):
    """
    Two Layer Neural Network.
    """

    def __init__(self, input_n, hidden_n, output_n, activation_fn):
        """
        Initialize TwoLayerNN.
        :param input_n: number of inputs
        :param hidden_n: number of hidden neurons
        :param output_n: number of outputs
        :param activation_fn: activation function for the hidden layer
        """
        super(TwoLayerNN, self).__init__()
        print("\n*** TwoLayerNN i: %s - h: %s - o: %s ***" % (input_n, hidden_n, output_n))
        self.hidden_linear = nn.Linear(input_n, hidden_n)
        self.hidden_activation = activation_fn
        self.output_linear = nn.Linear(hidden_n, output_n)
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        """
        Pass the input through the NN layers.
        :param input: input to the module
        :return: output from the module
        """
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return self.softmax(output_t)


def train_nn(iterations, nn_model, optimizer, nn_loss_fn, tensor_x, tensor_y, input_n, output_n):
    """
    Train Neural Network.
    :param iterations: epochs
    :param nn_model: NN model
    :param optimizer: optimizer
    :param nn_loss_fn: loss function
    :param tensor_x: X tensor X
    :param tensor_y: Y tensor
    :param input_n: number of inputs
    :param output_n: number of outputs
    :return:
    """
    print("\n*** TRAINING NN ***")
    print("\ntensor_x (%s): %s" % (tensor_x.shape, tensor_x))
    print("\ntensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    tensor_x_reshaped = tensor_x.view(-1, input_n)
    print("\ntensor_x_reshaped (%s): %s" % (tensor_x_reshaped.shape, tensor_x_reshaped))

    for it in range(1, iterations + 1):
        tensor_y_pred = nn_model(tensor_x_reshaped)
        loss_output = nn_loss_fn(tensor_y_pred, tensor_y)

        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        if it % 100 == 0:
            print("N: %s\t | Loss: %f\t" % (it, loss_output))


def main(trainingdataf="train-images.idx3-ubyte", traininglabelf="train-labels.idx1-ubyte",
         testdataf="t10k-images.idx3-ubyte", testlabelf="t10k-labels.idx1-ubyte"):
    # --------------- TRAINING ---------------
    # read training data
    train_data, train_data_dims = read_idx.read(trainingdataf, 50)
    train_labels, train_label_dims = read_idx.read(traininglabelf, 50)

    # convert tensors to the appropriate data types, also shape and normalize images data
    train_data = torch.tensor(train_data, dtype=torch.float)
    train_data = train_data.view((-1, 28 * 28))
    train_data /= 255
    print("train_data (%s): %s" % (train_data.size(), train_data))

    train_labels = torch.tensor(train_labels).long()
    print("train_labels (%s): %s" % (train_labels.size(), train_labels))

    '''
    # Filter training data by label: train_labels == 2 will return a tensor with True/False depending on the label for each sample
    # this True/False tensor can be used to index train_data, returning only the ones for which the condition was True
    twos = train_data[train_labels == 2]
    print("twos (%s): %s" % (twos.size(), twos))
    # show the first "2" on the screen
    lab3.show_image(twos[0], scale=lab3.SCALE_01)
    '''
    # NOTE FOR LAB: 30 neurons work very well
    model = TwoLayerNN(28*28, 30, 10, nn.ReLU())
    # NOTE FOR LAB: 1e-2 works better than 1e-2
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    print("\nmodel: %s" % model)

    train_nn(1000, model, optimizer, loss_fn, train_data, train_labels, 28*28, 10)
    train_y_pred = model(train_data)
    train_predictions = train_y_pred.max(1).indices

    print("\ntrain_y_pred (%s): %s ..." % (train_y_pred.size(), train_y_pred[:10, :]))
    print("\ntrain_labels (%s): %s" % (len(train_labels), train_labels))
    print("train_predic (%s): %s" % (len(train_predictions), train_predictions))

    # --------------- VALIDATION ---------------
    # read test data
    test_data, test_data_dims = read_idx.read(testdataf, 50)
    test_labels, test_label_dims = read_idx.read(testlabelf, 50)

    # convert tensors to the appropriate data types, also shape and normalize images data
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_data = test_data.view((-1, 28 * 28))
    # normalize training data
    test_data /= 255
    print("'\ntest_data (%s): %s" % (test_data.size(), test_data))

    test_labels = torch.tensor(test_labels).long()
    print("test_labels (%s): %s" % (test_labels.size(), test_labels))

    # predict test labels using trained model
    test_y_pred = model(test_data)
    test_predictions = test_y_pred.max(1).indices

    print("\ntest_y_pred (%s): %s ..." % (test_y_pred.size(), test_y_pred[:10, :]))
    print("\ntest_labels (%s): %s" % (len(test_labels), test_labels))
    print("test_predic (%s): %s" % (len(test_predictions), test_predictions))


if __name__ == "__main__":
    main()
