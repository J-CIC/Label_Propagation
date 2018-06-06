import pandas as pd
import numpy as np
import random
from scipy.misc import comb


def cal_distance(vec1,vec2):
    return np.sum(np.square(vec1 -vec2))

def generate_weight_graph(graph_array,max_id,a):
    weight_graph = np.zeros((max_id,max_id))
    for i in range(0,max_id):
        for j in range(i,max_id):
            weight_graph[i][j] = np.e**(-cal_distance(graph_array[i],graph_array[j])/(a*a))
            weight_graph[j][i] = weight_graph[i][j]
    return weight_graph

edge_link = pd.read_table("email-Eu-core.txt",' ',header=None)
labels_res = pd.read_table("email-Eu-core-department-labels.txt"," ",header=None)
# label_class = 42 # from 0 to 41
max_id = max(edge_link[0].max(),edge_link[1].max()) + 1
label_class = len(labels_res[1].unique())

graph_array = np.zeros((max_id, max_id))
for index,row in edge_link.iterrows():
    graph_array[row[0]][row[1]] = 1
    graph_array[row[1]][row[0]] = 1

weight_graph = generate_weight_graph(graph_array,max_id,1)
for i in range(0,max_id):
    weight_graph[i] = weight_graph[i]/weight_graph[i].sum()


def loop(frac):
    sum = 0.0
    for i in range(0,10):
        sum = sum + main(frac)
    print("Average result of frac %f is %f "% (frac,sum/10))

def main(frac):
    train=labels_res.sample(frac=frac)
    test=labels_res.drop(train.index)

    matrix = np.zeros((max_id,label_class))
    for index,row in train.iterrows():
        matrix[index][row[1]] = 1

    for index,row in test.iterrows():
        matrix[index][random.randint(0,label_class-1)] = 1
    iter_count = 0
    while(True):
        label_true=list()
        label_predict = list()
        t_matrix = np.dot(weight_graph,matrix)
        count = 0
        for index,row in train.iterrows():
            t_matrix[index].fill(0)
            t_matrix[index][row[1]] = 1

        for index,row in test.iterrows():
            idx = t_matrix[index].argmax()
            idx2 = matrix[index].argmax()
            label_true.append(row[1])
            label_predict.append(idx)
            t_matrix[index].fill(0)
            t_matrix[index][idx] = 1
            if(idx!=idx2):
                count = count + 1

        matrix = t_matrix
        iter_count = iter_count +1
        # print("iter %d:"%iter_count, " diff count:",count)
        if(count==0):
            break
    result = rand_index_score(label_true,label_predict)
    # print("Converge after %d iterations" %iter_count," Final Rand index score of frac %f"%frac,result)
    return result
# print(label_true)
# print(label_predict)
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)





if __name__ == '__main__':
    loop(0.30)
    loop(0.35)
    loop(0.40)
    loop(0.45)
    loop(0.50)
    loop(0.55)
    loop(0.60)
    loop(0.65)
    loop(0.70)
    loop(0.75)
    loop(0.80)
