import csv
import math
import random
import json
import sys

def test_network(data_file):
    neural_network = None
    #load neural network
    with open(data_file) as f:
        neural_network = json.load(f)

    data = []
    #load data
    with open("test_data.csv", newline="") as f:
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
            row_index += 1

    targets = []
    #load labels
    with open("test_labels.csv", newline="") as f:
        reader = csv.reader(f)
        row_index = 0
        header = 1
        for row in reader:
            if header == 1:
                header = 0
                continue
            targets.append([])
            for col in range(0, len(row)):
                targets[row_index] = (int(row[col]))
            row_index += 1

    #load values needed from neural network
    zero_index = neural_network["zero_index"]
    mean_list = neural_network["mean_list"]
    variance_list = neural_network["variance_list"]
    INPUT_COUNT = neural_network["input_count"]
    HIDDEN_COUNT = neural_network["hidden_count"]
    OUTPUT_COUNT = neural_network["output_count"]
    input_weights = neural_network["input_weights"]
    hidden_weights = neural_network["hidden_weights"]
    bias_weights = neural_network["bias_weights"]

    #delete useless nodes
    for row in range(0, len(data)):
        for index in reversed(zero_index):
            del data[row][index]

    #normalize
    for row in data:
        for i in range(0, INPUT_COUNT):
            row[i] = (row[i] - mean_list[i]) / variance_list[i]

    #initialize hidden nodes
    hidden_nodes = []
    for i in range(0, HIDDEN_COUNT):
        hidden_nodes.append(0)

    #initialize output nodes
    output_nodes = []
    for i in range(0, OUTPUT_COUNT):
        output_nodes.append(0)

    bias_node = -1

    #calculate number correct
    num_correct = 0
    for row_index in range(0, len(data)):
        row = data[row_index]
        for h_node in range(0, HIDDEN_COUNT):
            h = 0.0
            for i in range(0, INPUT_COUNT):
                h += input_weights[h_node][i] * row[i]
            h += bias_weights[0][h_node] * bias_node
            hidden_nodes[h_node] = 1 / (1 + math.exp(-h))
        for o_node in range(0, len(output_nodes)):
            h = 0.0
            for i in range(0, HIDDEN_COUNT):
                h += hidden_weights[o_node][i] * hidden_nodes[i]
            h += bias_weights[1][o_node] * bias_node
            output_nodes[o_node] = 1 / (1 + math.exp(-h))
        guess = compute_guess(output_nodes)

        if guess == targets[row_index]:
            num_correct += 1

    print("number correct:\t%d\t(%f%%)" % (num_correct, num_correct / 2000 * 100))

def compute_guess(output_nodes):
    max_output = max(output_nodes)
    for i in range(0, len(output_nodes)):
        if max_output == output_nodes[i]:
            return i

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n\tpython3 testNetwork.py <filename.data>")
        sys.exit(1)
    test_network(sys.argv[1])