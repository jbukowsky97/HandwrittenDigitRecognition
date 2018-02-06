import csv
import math
import random
import json

HIDDEN_COUNT = 20
OUTPUT_COUNT = 10
LEARNING_RATE = 0.1
VALIDATION_SIZE = 1000

data = []
min_list = []
max_list = []
mean_list = []
for i in range(0, 784):
    min_list.append(256)
    max_list.append(-1)
with open('train.csv', newline='') as f:
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

zero_index = []
for i in range(0, 784):
    if max_list[i] - min_list[i] == 0.0:
        zero_index.append(i)

INPUT_COUNT = 784 - len(zero_index)

for index in reversed(zero_index):
    del min_list[index]
    del max_list[index]
#del useless nodes
for row in range(0, len(data)):
    for index in reversed(zero_index):
        del data[row][index + 1]
    # for col in range(0, INPUT_COUNT):
    #     data[row][col + 1] = (data[row][col + 1] - min_list[col]) / (max_list[col] - min_list[col])
#normalize data

for i in range(0, INPUT_COUNT):
    mean_list.append(0)
for row in data:
    for col in range(0, INPUT_COUNT):
        mean_list[col] += row[col + 1]

mean_list = [x / 40000 for x in mean_list]
variance = []
for i in range(0, INPUT_COUNT):
    variance.append(0)
    for row in data:
        variance[i] += (row[i + 1] - mean_list[i]) ** 2
    variance[i] /= 40000
    variance[i] **= 0.5

for row in data:
    for i in range(0, INPUT_COUNT):
        row[i + 1] = (row[i + 1] - mean_list[i]) / variance[i]

input_weights = []
for row in range(0, HIDDEN_COUNT):
    input_weights.append([])
    for i in range(0, 784):
        input_weights[row].append(random.uniform(-0.3, 0.3))

hidden_nodes = []
for i in range(0, HIDDEN_COUNT):
    hidden_nodes.append(0)
#hidden_nodes.append(-1)

hidden_weights = []
for row in range(0, 10):
    hidden_weights.append([])
    for i in range(0, HIDDEN_COUNT):
        hidden_weights[row].append(random.uniform(-0.3, 0.3))
    
output_nodes = []
for i in range(0, 10):
    output_nodes.append(0)

bias_node = -1

bias_weights = []
bias_weights.append([])
bias_weights.append([])
for i in range(0, HIDDEN_COUNT):
    bias_weights[0].append(random.uniform(-0.1, 0.1))
for i in range(0, 10):
    bias_weights[1].append(random.uniform(-0.1, 0.1))

#TO-DO change while to depend on accuracy of validation
for epoch in range(0, 20):

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
        guess = 0
        if max(output_nodes) == output_nodes[0]:
            guess = 0
        elif max(output_nodes) == output_nodes[1]:
            guess = 1
        elif max(output_nodes) == output_nodes[2]:
            guess = 2
        elif max(output_nodes) == output_nodes[3]:
            guess = 3
        elif max(output_nodes) == output_nodes[4]:
            guess = 4
        elif max(output_nodes) == output_nodes[5]:
            guess = 5
        elif max(output_nodes) == output_nodes[6]:
            guess = 6
        elif max(output_nodes) == output_nodes[7]:
            guess = 7
        elif max(output_nodes) == output_nodes[8]:
            guess = 8
        elif max(output_nodes) == output_nodes[9]:
            guess = 9

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
    averages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    true_positives = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    actuals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wrong = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    avg_h = 0.0
    count_h = 0.0
    avg_hidden_h = 0.0
    count_hidden_h = 0
    avg_sigmoid = 0.0
    count1 = 0
    avg_sigmoid_hidden = 0.0
    count2 = 0
    for row in validation_set:
        for h_node in range(0, HIDDEN_COUNT):
            h = 0.0
            for i in range(0, INPUT_COUNT):
                h += input_weights[h_node][i] * row[i + 1]
            h += bias_weights[0][h_node] * bias_node
            hidden_nodes[h_node] = 1 / (1 + math.exp(-h))
            avg_sigmoid += 1 / (1 + math.exp(-h))
            count1 += 1
            avg_h += h
            count_h += 1
        for o_node in range(0, len(output_nodes)):
            h = 0.0
            for i in range(0, HIDDEN_COUNT):
                h += hidden_weights[o_node][i] * hidden_nodes[i]
            h += bias_weights[1][o_node] * bias_node
            output_nodes[o_node] = 1 / (1 + math.exp(-h))
            avg_sigmoid_hidden += 1 / (1 + math.exp(-h))
            count2 += 1
            avg_hidden_h += h
            count_hidden_h += 1
        guess = 0
        if max(output_nodes) == output_nodes[0]:
            guess = 0
        elif max(output_nodes) == output_nodes[1]:
            guess = 1
        elif max(output_nodes) == output_nodes[2]:
            guess = 2
        elif max(output_nodes) == output_nodes[3]:
            guess = 3
        elif max(output_nodes) == output_nodes[4]:
            guess = 4
        elif max(output_nodes) == output_nodes[5]:
            guess = 5
        elif max(output_nodes) == output_nodes[6]:
            guess = 6
        elif max(output_nodes) == output_nodes[7]:
            guess = 7
        elif max(output_nodes) == output_nodes[8]:
            guess = 8
        elif max(output_nodes) == output_nodes[9]:
            guess = 9

        if guess == row[0]:
            true_positives[guess] += 1
            num_correct += 1
        else:
            wrong[guess] += 1
        actuals[row[0]] += 1

        for i in range(0, 10):
            averages[i] += output_nodes[i]

    print("number correct:\t%d\t(%f%%)" % (num_correct, num_correct / VALIDATION_SIZE * 100))
    print(true_positives)
    print(wrong)
    print(actuals)
    for i in range(0, 10):
        averages[i] /= 1000
    print(averages)
    print(avg_sigmoid / count1)
    print(avg_h / count_h)
    print(avg_sigmoid_hidden / count2)
    print(avg_hidden_h / count_hidden_h)
    ###################################################

    print("saving")
    with open("epoch%d.data" % (epoch), "w") as wfile:
        json.dump({"zero_index": zero_index, "mean_list": mean_list, "variance_list": variance, "input_count": INPUT_COUNT, "hidden_count": HIDDEN_COUNT, "output_count": OUTPUT_COUNT, "input_weights": input_weights, "hidden_weights": hidden_weights, "bias_weights": bias_weights}, wfile)
    print("epoch %d complete" % (epoch))


print("done")