import csv
import math
import random
import json

HIDDEN_COUNT = 20
OUTPUT_COUNT = 10
LEARNING_RATE = 0.1
VALIDATION_SIZE = 1000

def train_network():

    data = []
    min_list = []
    max_list = []
    mean_list = []

    #initialize values for min and max lists
    for i in range(0, 784):
        min_list.append(256)
        max_list.append(-1)

    #read csv file into two-dimensional data array
    with open("train.csv", newline="") as f:
        reader = csv.reader(f)
        row_index = 0
        header = 1
        for row in reader:
            if header == 1:
                header = 0
                continue
            data.append([])
            for col in range(0, len(row)):
                data[row_index].append(int(row[col]))
                if col > 0:
                    if int(row[col]) < min_list[col - 1]:
                        min_list[col - 1] = int(row[col])
                    if int(row[col]) > max_list[col - 1]:
                        max_list[col - 1] = int(row[col])
            row_index += 1

    #find nodes that do not change
    zero_index = []
    for i in range(0, 784):
        if max_list[i] - min_list[i] == 0.0:
            zero_index.append(i)

    #calculate input count disregarding useless nodes
    INPUT_COUNT = 784 - len(zero_index)

    #delete no-longer used stats
    del min_list
    del max_list

    #del useless nodes
    for row in range(0, len(data)):
        for index in reversed(zero_index):
            del data[row][index + 1]

    #calculate means
    for i in range(0, INPUT_COUNT):
        mean_list.append(0)
    for row in data:
        for col in range(0, INPUT_COUNT):
            mean_list[col] += row[col + 1]

    mean_list = [x / 40000 for x in mean_list]

    #calculate variances (actually deviation)
    variance = []
    for i in range(0, INPUT_COUNT):
        variance.append(0)
        for row in data:
            variance[i] += (row[i + 1] - mean_list[i]) ** 2
        variance[i] /= 40000
        variance[i] **= 0.5

    #normalize data
    for row in data:
        for i in range(0, INPUT_COUNT):
            row[i + 1] = (row[i + 1] - mean_list[i]) / variance[i]

    #initialize hidden weights
    input_weights = []
    for row in range(0, HIDDEN_COUNT):
        input_weights.append([])
        for i in range(0, 784):
            input_weights[row].append(random.uniform(-0.3, 0.3))

    #initialize hidden nodes
    hidden_nodes = []
    for i in range(0, HIDDEN_COUNT):
        hidden_nodes.append(0)
    
    #initialize hidden weights
    hidden_weights = []
    for row in range(0, 10):
        hidden_weights.append([])
        for i in range(0, HIDDEN_COUNT):
            hidden_weights[row].append(random.uniform(-0.3, 0.3))
    
    #initialize output nodes
    output_nodes = []
    for i in range(0, 10):
        output_nodes.append(0)

    bias_node = -1

    #initialize bias weights
    bias_weights = []
    bias_weights.append([])
    bias_weights.append([])
    for i in range(0, HIDDEN_COUNT):
        bias_weights[0].append(random.uniform(-0.1, 0.1))
    for i in range(0, 10):
        bias_weights[1].append(random.uniform(-0.1, 0.1))


    for epoch in range(0, 100):

        random.shuffle(data)
        training_set = data[0:-VALIDATION_SIZE]
        validation_set = data[-VALIDATION_SIZE:]

        print("training")
        for row in training_set:
            for h_node in range(0, HIDDEN_COUNT):
                h = 0.0
                for i in range(0, INPUT_COUNT):
                    h += input_weights[h_node][i] * row[i + 1]
                h += bias_weights[0][h_node] * bias_node
                hidden_nodes[h_node] = 1 / (1 + math.exp(-h))
            for o_node in range(0, OUTPUT_COUNT):
                h = 0.0
                for i in range(0, HIDDEN_COUNT):
                    h += hidden_weights[o_node][i] * hidden_nodes[i]
                h += bias_weights[1][o_node] * bias_node
                output_nodes[o_node] = 1 / (1 + math.exp(-h))
            guess = compute_guess(output_nodes)

            if guess != row[0]:
                error_output = []
                error_hidden = []
                for o_node in range(0, len(output_nodes)):
                    target = 0
                    if o_node == row[0]:
                        target = 1
                    else:
                        target = 0
                    error_output.append(output_nodes[o_node] * (1 - output_nodes[o_node]) * (target - output_nodes[o_node]))
                for h_node in range(0, HIDDEN_COUNT):
                    sum_errors = 0.0
                    for i in range(0, 10):
                        sum_errors += hidden_weights[i][h_node] * error_output[i]
                    error_hidden.append(hidden_nodes[h_node] * (1 - hidden_nodes[h_node]) * sum_errors)
                for o_node in range(0, 10):
                    for h_node in range(0, HIDDEN_COUNT):
                        hidden_weights[o_node][h_node] += LEARNING_RATE * error_output[o_node] * hidden_nodes[h_node]
                    bias_weights[1][o_node] += LEARNING_RATE * error_output[o_node] * bias_node
                for h_node in range(0, HIDDEN_COUNT):
                    for i_node in range(0, INPUT_COUNT):
                        input_weights[h_node][i_node] += LEARNING_RATE * error_hidden[h_node] * row[i_node + 1]
                    bias_weights[0][h_node] += LEARNING_RATE * error_hidden[h_node] * bias_node

        ###################################################
        print("validating")
        num_correct = 0
        for row in validation_set:
            for h_node in range(0, HIDDEN_COUNT):
                h = 0.0
                for i in range(0, INPUT_COUNT):
                    h += input_weights[h_node][i] * row[i + 1]
                h += bias_weights[0][h_node] * bias_node
                hidden_nodes[h_node] = 1 / (1 + math.exp(-h))
            for o_node in range(0, len(output_nodes)):
                h = 0.0
                for i in range(0, HIDDEN_COUNT):
                    h += hidden_weights[o_node][i] * hidden_nodes[i]
                h += bias_weights[1][o_node] * bias_node
                output_nodes[o_node] = 1 / (1 + math.exp(-h))

            guess = compute_guess(output_nodes)

            if guess == row[0]:
                num_correct += 1

        print("number correct:\t%d\t(%f%%)" % (num_correct, num_correct / VALIDATION_SIZE * 100))
        ###################################################

        print("saving")
        with open("epochs/epoch%d.data" % (epoch), "w") as wfile:
            json.dump({"zero_index": zero_index, "mean_list": mean_list, "variance_list": variance, "input_count": INPUT_COUNT, "hidden_count": HIDDEN_COUNT, "output_count": OUTPUT_COUNT, "input_weights": input_weights, "hidden_weights": hidden_weights, "bias_weights": bias_weights}, wfile)
        print("epoch %d complete" % (epoch))
    print("done")

def compute_guess(output_nodes):
    max_output = max(output_nodes)
    for i in range(0, len(output_nodes)):
        if max_output == output_nodes[i]:
            return i

if __name__ == "__main__":
    train_network()