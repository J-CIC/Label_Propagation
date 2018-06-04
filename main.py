import pandas as pd
import numpy as np
import random


def cal_distance(vec1,vec2):
    return np.sum(np.square(vec1 -vec2))

def generate_weight_graph(graph_array,max_id,a):
    weight_graph = np.zeros((max_id,max_id))
    for i in range(0,max_id):
        for j in range(i+1,max_id):
            weight_graph[i][j] = cal_distance(graph_array[i],graph_array[j])/(a*a)
    return weight_graph

def main():
    pass
edge_link = pd.read_table("email-Eu-core.txt",' ',header=None)
labels_res = pd.read_table("email-Eu-core-department-labels.txt"," ",header=None)
# label_class = 42 # from 0 to 41
max_id = max(edge_link[0].max(),edge_link[1].max()) + 1
label_class = len(labels_res[1].unique())

graph_array = np.zeros((max_id, max_id))
for index,row in edge_link.iterrows():
    graph_array[row[0]][row[1]] = 1
    graph_array[row[1]][row[0]] = 1

weight_graph = generate_weight_graph(graph_array,max_id,2)
avg_vec = np.average(weight_graph,axis=0)
# new_matrix = np.zeros((max_id,max_id))
for i in range(0,max_id):
    weight_graph[i] = weight_graph[i]/weight_graph[i].sum()

train=labels_res.sample(frac=0.05,random_state=200)
test=labels_res.drop(train.index)

for index,row in test.iterrows():
    row[1] = random.randint(0,label_class-1)

matrix = np.zeros((max_id,label_class))
for index,row in train.iterrows():
    matrix[index][row[1]] = 1

for index,row in test.iterrows():
    matrix[index][row[1]] = 1

t_matrix = np.dot(weight_graph,matrix)
#todo 

if __name__ == '__main__':
    main()
