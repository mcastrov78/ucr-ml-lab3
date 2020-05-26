import read_idx
import lab3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# None to process all images
NUMBER_OF_IMAGES_TO_PROCESS = 500


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
    tensor_x_reshaped = tensor_x.view(-1, input_n)
    #print("\ntensor_x_reshaped (%s): %s" % (tensor_x_reshaped.shape, tensor_x_reshaped))
    #print("\ntensor_y (%s): %s" % (tensor_y.shape, tensor_y))

    for it in range(1, iterations + 1):
        tensor_y_pred = nn_model(tensor_x_reshaped)
        loss_output = nn_loss_fn(tensor_y_pred, tensor_y)

        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        if it % 100 == 0:
            print("N: %s\t | Loss: %f\t" % (it, loss_output))


def get_images_and_labels_tensors(images_filename, labels_filename):
    images_tensor, images_data_dims = read_idx.read(images_filename, NUMBER_OF_IMAGES_TO_PROCESS)
    labels_tensor, labels_data_dims = read_idx.read(labels_filename, NUMBER_OF_IMAGES_TO_PROCESS)

    # convert tensors to the appropriate data types, also shape and normalize images data
    images_tensor = torch.tensor(images_tensor, dtype=torch.float)
    images_tensor = images_tensor.view((-1, 28 * 28))
    images_tensor /= 255

    labels_tensor = torch.tensor(labels_tensor).long()
    #print("images_tensor (%s): %s" % (images_tensor.size(), images_tensor))
    #print("labels_tensor (%s): %s" % (labels_tensor.size(), labels_tensor))

    return images_tensor, labels_tensor


def print_digit(image_tensor, label):
    print("\nlabel: %s" % (label))
    this_image_tensor = (image_tensor * 255).type(torch.int)
    print("data: %s" % (this_image_tensor.view(28, 28)))
    lab3.show_image(this_image_tensor, "sample_images\{}.png".format(label), scale=lab3.SCALE_OFF)


def train(trainingdataf, traininglabelf, model, learning_rate):
    print("\n--------------- TRAINING - --------------")
    # read training data
    train_data, train_labels = get_images_and_labels_tensors(trainingdataf, traininglabelf)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    print("\nmodel: %s" % model)

    train_nn(1000, model, optimizer, loss_fn, train_data, train_labels, 28 * 28, 10)
    train_y_pred = model(train_data)
    train_predictions = train_y_pred.max(1).indices

    # print sample image tensors data along with its label and the image itself
    for i in range(3):
        print_digit(train_data[i], train_labels[i])

    #print("\ntrain_y_pred (%s): %s ..." % (train_y_pred.size(), train_y_pred[:10, :]))
    print("\ntrain_labels (%s): %s ..." % (len(train_labels), train_labels[:100]))
    print("train_predic (%s): %s ..." % (len(train_predictions), train_predictions[:100]))

    return train_data, train_labels


def generate_classification_report(test_labels, test_predictions):
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    print("\nconf_matrix (%s): \n%s" % (len(conf_matrix), conf_matrix))

    precision = precision_score(test_labels, test_predictions, average=None)
    print("\nprecision (%s): \n%s" % (len(precision), precision))

    recall = recall_score(test_labels, test_predictions, average=None)
    print("\nrecall (%s): \n%s" % (len(recall), recall))

    clasif_report = classification_report(test_labels, test_predictions)
    print("\nclasif_report (%s): \n%s" % (len(clasif_report), clasif_report))


def validate(testdataf, testlabelf, model):
    print("\n--------------- VALIDATION - --------------")
    # read test data
    test_data, test_labels = get_images_and_labels_tensors(testdataf, testlabelf)

    # predict test labels using trained model
    test_y_pred = model(test_data)
    test_predictions = test_y_pred.max(1).indices

    #print("\ntest_y_pred (%s): %s ..." % (test_y_pred.size(), test_y_pred[:10, :]))
    print("\ntest_labels (%s): %s ..." % (len(test_labels), test_labels[:100]))
    print("test_predic (%s): %s ..." % (len(test_predictions), test_predictions[:100]))

    generate_classification_report(test_labels, test_predictions)

    return test_data, test_labels


def analyze_weights(model):
    print("model: %s" % model)
    named_parameters = model.named_parameters()
    print("\nNAMED PARAMETERS: %s" % named_parameters)
    for name, param, in named_parameters:
        print("\nname: %s" % name)
        print("tensor(%s): %s" % (param.size(), param.data))
        if name == "hidden_linear.weight":
            for i in range(param.size()[0]):
                this_image_tensor = (param.data[i] * 255).type(torch.int)
                #print("\ndata(%s): %s" % (i, this_image_tensor.view(28, 28)))
                lab3.show_image(this_image_tensor, "neuron_weight_images/{}.png".format(i), scale=lab3.SCALE_OFF)


def main(trainingdataf="train-images.idx3-ubyte", traininglabelf="train-labels.idx1-ubyte",
         testdataf="t10k-images.idx3-ubyte", testlabelf="t10k-labels.idx1-ubyte"):

    # we want to see tensor rows in a single line
    torch.set_printoptions(linewidth=300)

    # --------------- TRAINING ---------------
    # NOTE FOR LAB: 50 neurons works the best. More than that doesn't really improve.
    # NOTE FOR LAB: 1e-3 works better than 1e-2 and 1e-2. 1e-4 can cause errors
    model = TwoLayerNN(28 * 28, 50, 10, nn.Sigmoid())
    train(trainingdataf, traininglabelf, model, 1e-3)
    validate(testdataf, testlabelf, model)

    # --------------- TRAINING ---------------
    model = TwoLayerNN(28 * 28, 50, 10, nn.ReLU())
    train(trainingdataf, traininglabelf, model, 1e-3)
    validate(testdataf, testlabelf, model)

    analyze_weights(model)


if __name__ == "__main__":
    main()
