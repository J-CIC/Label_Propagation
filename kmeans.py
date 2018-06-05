from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import comb

edge_link = pd.read_table("email-Eu-core.txt",' ',header=None)
labels_res = pd.read_table("email-Eu-core-department-labels.txt"," ",header=None)
# label_class = 42 # from 0 to 41
max_id = max(edge_link[0].max(),edge_link[1].max()) + 1
label_class = len(labels_res[1].unique())

graph_array = np.zeros((max_id, max_id))
for index,row in edge_link.iterrows():
    graph_array[row[0]][row[1]] = 1
    graph_array[row[1]][row[0]] = 1

kmeans = KMeans(n_clusters=label_class).fit(graph_array)
kmeans.labels_

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

print(rand_index_score(kmeans.labels_,labels_res[1]))