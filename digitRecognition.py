import csv
import math
import random

HIDDEN_COUNT = 100
OUTPUT_COUNT = 10
LEARNING_RATE = 0.9

data = []
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
            if col == 0:
                data[row_index].append(int(row[col]))
            else:
                data[row_index].append(int(row[col]) / 127.5 - 1)
        #data[row_index].append(-1)
        row_index += 1

random.shuffle(data)
# sets 0 - 6 for training, 7 only for validation
training_set = []

relative_index = 0
max_index = len(data) // 8
for row in range(relative_index, 40000):
    relative_index += 1
    if row % 5000 == 0:
        training_set.append([])
    training_set[row // 5000].append([])
    training_set[row // 5000][row % 5000] = data[row]

input_weights = []
for row in range(0, 100):
    input_weights.append([])
    for i in range(0, 784):
        input_weights[row].append(random.uniform(-0.5, 0.5))

hidden_nodes = []
for i in range(0, 100):
    hidden_nodes.append(0)
#hidden_nodes.append(-1)

hidden_weights = []
for row in range(0, 10):
    hidden_weights.append([])
    for i in range(0, 100):
        hidden_weights[row].append(random.uniform(-0.5, 0.5))
    
output_nodes = []
for i in range(0, 10):
    output_nodes.append(0)

for data_set in range(0, 6):
    print("validating")
    num_correct = 0
    validation = training_set[7]
    for row in validation:
        for h_node in range(0, len(hidden_nodes)):
            h = 0.0
            for i in range(0, len(row) - 1):
                h += input_weights[h_node][i] * row[i + 1]
            hidden_nodes[h_node] = 1 / (1 + math.exp(h))
        for o_node in range(0, len(output_nodes)):
            h = 0.0
            for i in range(0, len(hidden_nodes)):
                h += hidden_weights[o_node][i] * hidden_nodes[i]
            output_nodes[o_node] = 1 / (1 + math.exp(h))
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
            num_correct += 1

    print("number correct:\t%d\t(%f)" % (num_correct, num_correct / 5000 * 100))
    ###################################################
    print("training")
    cur_set = training_set[data_set]
    for row in cur_set:
        for h_node in range(0, len(hidden_nodes)):
            h = 0.0
            for i in range(0, len(row) - 1):
                h += input_weights[h_node][i] * row[i + 1]
            hidden_nodes[h_node] = 1 / (1 + math.exp(h))
        for o_node in range(0, len(output_nodes)):
            h = 0.0
            for i in range(0, len(hidden_nodes)):
                h += hidden_weights[o_node][i] * hidden_nodes[i]
            output_nodes[o_node] = 1 / (1 + math.exp(h))
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
                if o_node == guess:
                    target = 1
                else:
                    target = -1
                error_output.append(output_nodes[o_node] * (1 - output_nodes[o_node]) * (target - output_nodes[o_node]))
            for h_node in range(0, len(hidden_nodes)):
                sum_errors = 0.0
                for i in range(0, 10):
                    sum_errors += hidden_weights[i][h_node] * error_output[i]
                error_hidden.append(hidden_nodes[h_node] * (1 - hidden_nodes[h_node]) * sum_errors)
            for o_node in range(0, 10):
                for h_node in range(0, 100):
                    hidden_weights[o_node][h_node] += LEARNING_RATE * error_output[o_node] * hidden_nodes[h_node]
            for h_node in range(0, 100):
                for i_node in range(0, 784):
                    input_weights[h_node][i_node] += LEARNING_RATE * error_hidden[h_node] * row[i_node + 1]


print("done")